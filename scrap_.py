import os
import time

import json
import unicodedata
import os
import re

from bs4 import BeautifulSoup
import requests
import torch
import torchaudio
# import sox
import json
import argparse
import soundfile as sf
import soundfile
import subprocess

import torch
import tempfile
import math
from dataclasses import dataclass
from torchaudio.models import wav2vec2_model
import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# iso codes with specialized rules in uroman
special_isos_uroman = "ara, bel, bul, deu, ell, eng, fas, grc, ell, eng, heb, kaz, kir, lav, lit, mkd, mkd2, oss, pnt, pus, rus, srp, srp2, tur, uig, ukr, yid".split(
    ",")
special_isos_uroman = [i.strip() for i in special_isos_uroman]


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def get_uroman_tokens(norm_transcripts, uroman_root_dir, iso=None):
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf2 = tempfile.NamedTemporaryFile(delete=False)
    with open(tf.name, "w", encoding='utf-8') as f:
        for t in norm_transcripts:
            f.write(t + "\n")

    assert os.path.exists(f"{uroman_root_dir}/uroman.pl"), "uroman not found"

    cmd = fr"{uroman_root_dir}/uroman.pl"

    if iso in special_isos_uroman:
        cmd += f" -l {iso} "

    cmd += f" < {tf.name} > {tf2.name}"
    print(cmd)

    tf2.close()
    tf.close()
    # os.system(cmd)
    subprocess.call(cmd, shell=True)
    print("1")
    outtexts = []
    print("1")
    with open(tf2.name, encoding='utf-8') as f:
        for line in f:
            line = " ".join(line.strip())
            line = re.sub(r"\s+", " ", line).strip()
            outtexts.append(line)
    print(len(outtexts))
    print(len(norm_transcripts))
    assert len(outtexts) == len(norm_transcripts)
    uromans = []
    for ot in outtexts:
        uromans.append(normalize_uroman(ot))
    return uromans


@dataclass
class Segment:
    label: str
    start: int
    end: int

    def __repr__(self):
        return f"{self.label}: [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, idx_to_token_map):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1] == path[i2]:
            i2 += 1
        segments.append(Segment(idx_to_token_map[path[i1]], i1, i2 - 1))
        i1 = i2
    return segments


def time_to_frame(time):
    stride_msec = 20
    frames_per_sec = 1000 / stride_msec
    return int(time * frames_per_sec)


def load_model_dict():
    model_path_name = "./alignment/tmp/ctc_alignment_mling_uroman_model.pt"

    print("Downloading model and dictionary...")
    if os.path.exists(model_path_name):
        print("Model path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/model.pt",
            model_path_name,
        )
        assert os.path.exists(model_path_name)
    state_dict = torch.load(model_path_name, map_location="cpu")

    model = wav2vec2_model(
        extractor_mode="layer_norm",
        extractor_conv_layer_config=[
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias=True,
        encoder_embed_dim=1024,
        encoder_projection_dropout=0.0,
        encoder_pos_conv_kernel=128,
        encoder_pos_conv_groups=16,
        encoder_num_layers=24,
        encoder_num_heads=16,
        encoder_attention_dropout=0.0,
        encoder_ff_interm_features=4096,
        encoder_ff_interm_dropout=0.1,
        encoder_dropout=0.0,
        encoder_layer_norm_first=True,
        encoder_layer_drop=0.1,
        aux_num_out=31,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dict_path_name = "./alignment/tmp/ctc_alignment_mling_uroman_model.dict"
    if os.path.exists(dict_path_name):
        print("Dictionary path already exists. Skipping downloading....")
    else:
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/mms/torchaudio/ctc_alignment_mling_uroman/dictionary.txt",
            dict_path_name,
        )
        assert os.path.exists(dict_path_name)
    dictionary = {}
    with open(dict_path_name) as f:
        dictionary = {l.strip(): i for i, l in enumerate(f.readlines())}

    return model, dictionary


def get_spans(tokens, segments):
    ltr_idx = 0
    tokens_idx = 0
    intervals = []
    start, end = (0, 0)
    sil = "<blank>"
    for (seg_idx, seg) in enumerate(segments):
        if (tokens_idx == len(tokens)):
            assert (seg_idx == len(segments) - 1)
            assert (seg.label == '<blank>')
            continue
        cur_token = tokens[tokens_idx].split(' ')
        ltr = cur_token[ltr_idx]
        if seg.label == "<blank>": continue
        assert (seg.label == ltr)
        if (ltr_idx) == 0: start = seg_idx
        if ltr_idx == len(cur_token) - 1:
            ltr_idx = 0
            tokens_idx += 1
            intervals.append((start, seg_idx))
            while tokens_idx < len(tokens) and len(tokens[tokens_idx]) == 0:
                intervals.append((seg_idx, seg_idx))
                tokens_idx += 1
        else:
            ltr_idx += 1
    spans = []
    for (idx, (start, end)) in enumerate(intervals):
        span = segments[start:end + 1]
        if start > 0:
            prev_seg = segments[start - 1]
            if prev_seg.label == sil:
                pad_start = prev_seg.start if (idx == 0) else int((prev_seg.start + prev_seg.end) / 2)
                span = [Segment(sil, pad_start, span[0].start)] + span
        if end + 1 < len(segments):
            next_seg = segments[end + 1]
            if next_seg.label == sil:
                pad_end = next_seg.end if (idx == len(intervals) - 1) else math.floor(
                    (next_seg.start + next_seg.end) / 2)
                span = span + [Segment(sil, span[-1].end, pad_end)]
        spans.append(span)
    return spans


colon = ":"
comma = ","
exclamation_mark = "!"
period = re.escape(".")
question_mark = re.escape("?")
semicolon = ";"

left_curly_bracket = "{"
right_curly_bracket = "}"
quotation_mark = '"'

basic_punc = (
        period
        + question_mark
        + comma
        + colon
        + exclamation_mark
        + left_curly_bracket
        + right_curly_bracket
)

# General punc unicode block (0x2000-0x206F)
zero_width_space = r"\u200B"
zero_width_nonjoiner = r"\u200C"
left_to_right_mark = r"\u200E"
right_to_left_mark = r"\u200F"
left_to_right_embedding = r"\u202A"
pop_directional_formatting = r"\u202C"

# Here are some commonly ill-typed versions of apostrophe
right_single_quotation_mark = r"\u2019"
left_single_quotation_mark = r"\u2018"

# Language specific definitions
# Spanish
inverted_exclamation_mark = r"\u00A1"
inverted_question_mark = r"\u00BF"

# Hindi
hindi_danda = u"\u0964"

# Egyptian Arabic
# arabic_percent = r"\u066A"
arabic_comma = r"\u060C"
arabic_question_mark = r"\u061F"
arabic_semicolon = r"\u061B"
arabic_diacritics = r"\u064B-\u0652"

arabic_subscript_alef_and_inverted_damma = r"\u0656-\u0657"

# Chinese
full_stop = r"\u3002"
full_comma = r"\uFF0C"
full_exclamation_mark = r"\uFF01"
full_question_mark = r"\uFF1F"
full_semicolon = r"\uFF1B"
full_colon = r"\uFF1A"
full_parentheses = r"\uFF08\uFF09"
quotation_mark_horizontal = r"\u300C-\u300F"
quotation_mark_vertical = r"\uFF41-\uFF44"
title_marks = r"\u3008-\u300B"
wavy_low_line = r"\uFE4F"
ellipsis = r"\u22EF"
enumeration_comma = r"\u3001"
hyphenation_point = r"\u2027"
forward_slash = r"\uFF0F"
wavy_dash = r"\uFF5E"
box_drawings_light_horizontal = r"\u2500"
fullwidth_low_line = r"\uFF3F"
chinese_punc = (
        full_stop
        + full_comma
        + full_exclamation_mark
        + full_question_mark
        + full_semicolon
        + full_colon
        + full_parentheses
        + quotation_mark_horizontal
        + quotation_mark_vertical
        + title_marks
        + wavy_low_line
        + ellipsis
        + enumeration_comma
        + hyphenation_point
        + forward_slash
        + wavy_dash
        + box_drawings_light_horizontal
        + fullwidth_low_line
)

# Armenian
armenian_apostrophe = r"\u055A"
emphasis_mark = r"\u055B"
exclamation_mark = r"\u055C"
armenian_comma = r"\u055D"
armenian_question_mark = r"\u055E"
abbreviation_mark = r"\u055F"
armenian_full_stop = r"\u0589"
armenian_punc = (
        armenian_apostrophe
        + emphasis_mark
        + exclamation_mark
        + armenian_comma
        + armenian_question_mark
        + abbreviation_mark
        + armenian_full_stop
)

lesser_than_symbol = r"&lt;"
greater_than_symbol = r"&gt;"

lesser_than_sign = r"\u003c"
greater_than_sign = r"\u003e"

nbsp_written_form = r"&nbsp"

# Quotation marks
left_double_quotes = r"\u201c"
right_double_quotes = r"\u201d"
left_double_angle = r"\u00ab"
right_double_angle = r"\u00bb"
left_single_angle = r"\u2039"
right_single_angle = r"\u203a"
low_double_quotes = r"\u201e"
low_single_quotes = r"\u201a"
high_double_quotes = r"\u201f"
high_single_quotes = r"\u201b"

all_punct_quotes = (
        left_double_quotes
        + right_double_quotes
        + left_double_angle
        + right_double_angle
        + left_single_angle
        + right_single_angle
        + low_double_quotes
        + low_single_quotes
        + high_double_quotes
        + high_single_quotes
        + right_single_quotation_mark
        + left_single_quotation_mark
)
mapping_quotes = (
        "["
        + high_single_quotes
        + right_single_quotation_mark
        + left_single_quotation_mark
        + "]"
)

# Digits

english_digits = r"\u0030-\u0039"
bengali_digits = r"\u09e6-\u09ef"
khmer_digits = r"\u17e0-\u17e9"
devanagari_digits = r"\u0966-\u096f"
oriya_digits = r"\u0b66-\u0b6f"
extended_arabic_indic_digits = r"\u06f0-\u06f9"
kayah_li_digits = r"\ua900-\ua909"
fullwidth_digits = r"\uff10-\uff19"
malayam_digits = r"\u0d66-\u0d6f"
myanmar_digits = r"\u1040-\u1049"
roman_numeral = r"\u2170-\u2179"
nominal_digit_shapes = r"\u206f"

# Load punctuations from MMS-lab data
with open("./punctuations.lst", "r", encoding='utf-8') as punc_f:
    punc_list = punc_f.readlines()

punct_pattern = r""
for punc in punc_list:
    # the first character in the tab separated line is the punc to be removed
    punct_pattern += re.escape(punc.split("\t")[0])

shared_digits = (
        english_digits
        + bengali_digits
        + khmer_digits
        + devanagari_digits
        + oriya_digits
        + extended_arabic_indic_digits
        + kayah_li_digits
        + fullwidth_digits
        + malayam_digits
        + myanmar_digits
        + roman_numeral
        + nominal_digit_shapes
)

shared_punc_list = (
        basic_punc
        + all_punct_quotes
        + greater_than_sign
        + lesser_than_sign
        + inverted_question_mark
        + full_stop
        + semicolon
        + armenian_punc
        + inverted_exclamation_mark
        + arabic_comma
        + enumeration_comma
        + hindi_danda
        + quotation_mark
        + arabic_semicolon
        + arabic_question_mark
        + chinese_punc
        + punct_pattern

)

shared_mappping = {
    lesser_than_symbol: "",
    greater_than_symbol: "",
    nbsp_written_form: "",
    r"(\S+)" + mapping_quotes + r"(\S+)": r"\1'\2",
}

shared_deletion_list = (
        left_to_right_mark
        + zero_width_nonjoiner
        + arabic_subscript_alef_and_inverted_damma
        + zero_width_space
        + arabic_diacritics
        + pop_directional_formatting
        + right_to_left_mark
        + left_to_right_embedding
)

norm_config = {
    "*": {
        "lower_case": True,
        "punc_set": shared_punc_list,
        "del_set": shared_deletion_list,
        "mapping": shared_mappping,
        "digit_set": shared_digits,
        "unicode_norm": "NFKC",
        "rm_diacritics": False,
    }
}

# =============== Mongolian ===============#

norm_config["mon"] = norm_config["*"].copy()
# add soft hyphen to punc list to match with fleurs
norm_config["mon"]["del_set"] += r"\u00AD"

norm_config["khk"] = norm_config["mon"].copy()

# =============== Hebrew ===============#

norm_config["heb"] = norm_config["*"].copy()
# add "HEBREW POINT" symbols to match with fleurs
norm_config["heb"]["del_set"] += r"\u05B0-\u05BF\u05C0-\u05CF"

# =============== Thai ===============#

norm_config["tha"] = norm_config["*"].copy()
# add "Zero width joiner" symbols to match with fleurs
norm_config["tha"]["punc_set"] += r"\u200D"

# =============== Arabic ===============#
norm_config["ara"] = norm_config["*"].copy()
norm_config["ara"]["mapping"]["ٱ"] = "ا"
norm_config["arb"] = norm_config["ara"].copy()

# =============== Javanese ===============#
norm_config["jav"] = norm_config["*"].copy()
norm_config["jav"]["rm_diacritics"] = True


def text_normalize(text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False):
    """Given a text, normalize it by changing to lower case, removing punctuations, removing words that only contain digits and removing extra spaces

    Args:
        text : The string to be normalized
        iso_code :
        remove_numbers : Boolean flag to specify if words containing only digits should be removed

    Returns:
        normalized_text : the string after all normalization

    """

    config = norm_config.get(iso_code, norm_config["*"])

    for field in ["lower_case", "punc_set", "del_set", "mapping", "digit_set", "unicode_norm"]:
        if field not in config:
            config[field] = norm_config["*"][field]

    text = unicodedata.normalize(config["unicode_norm"], text)

    # Convert to lower case

    if config["lower_case"] and lower_case:
        text = text.lower()

    # brackets

    # always text inside brackets with numbers in them. Usually corresponds to "(Sam 23:17)"
    text = re.sub(r"\([^\)]*\d[^\)]*\)", " ", text)
    if remove_brackets:
        text = re.sub(r"\([^\)]*\)", " ", text)

    # Apply mappings

    for old, new in config["mapping"].items():
        text = re.sub(old, new, text)

    # Replace punctutations with space

    punct_pattern = r"[" + config["punc_set"]

    punct_pattern += "]"

    normalized_text = re.sub(punct_pattern, " ", text)

    # remove characters in delete list

    delete_patten = r"[" + config["del_set"] + "]"

    normalized_text = re.sub(delete_patten, "", normalized_text)

    # Remove words containing only digits
    # We check for 3 cases  a)text starts with a number b) a number is present somewhere in the middle of the text c) the text ends with a number
    # For each case we use lookaround regex pattern to see if the digit pattern in preceded and followed by whitespaces, only then we replace the numbers with space
    # The lookaround enables overlapping pattern matches to be replaced

    if remove_numbers:
        digits_pattern = "[" + config["digit_set"]

        digits_pattern += "]+"

        complete_digit_pattern = (
                r"^"
                + digits_pattern
                + "(?=\s)|(?<=\s)"
                + digits_pattern
                + "(?=\s)|(?<=\s)"
                + digits_pattern
                + "$"
        )

        normalized_text = re.sub(complete_digit_pattern, " ", normalized_text)

    if config["rm_diacritics"]:
        from unidecode import unidecode
        normalized_text = unidecode(normalized_text)

    # Remove extra spaces
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    return normalized_text


def force(out, text_filepath, audio):
    outdir = out
    text_filepath = text_filepath
    uroman_path = './uroman/bin'
    lang = 'pus'
    # assert not os.path.exists(
    #     outdir
    # ), f"Error: Output path exists already {outdir}"

    transcripts = []
    with open(text_filepath, encoding='utf-8') as f:
        transcripts = [line.strip() for line in f]
    print("Read {} lines from {}".format(len(transcripts), text_filepath))

    norm_transcripts = [text_normalize(line.strip(), lang) for line in transcripts]

    tokens = get_uroman_tokens(norm_transcripts, uroman_path, lang)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    # if use_star:
    #     dictionary["<star>"] = len(dictionary)
    #     tokens = ["<star>"] + tokens
    #     transcripts = ["<star>"] + transcripts
    #     norm_transcripts = ["<star>"] + norm_transcripts

    # segments, stride = get_alignments(
    #     audio_filepath,
    #     tokens,
    #     model,
    #     dictionary,
    #     use_star,
    # )
    audio_file = audio
    tokens,
    model,
    dictionary,
    use_star = False,

    import pydub

    waveform, _ = torchaudio.load(audio_file)  # waveform: channels X T
    waveform = waveform.to(DEVICE)
    s = pydub.AudioSegment.from_file(audio_file)
    # sound,sr = sf.read(audio_file)
    total_duration = s.duration_seconds
    # total_duration = sox.file_info.duration(audio_file)
    print(f"Duration: {total_duration}")
    s = s.set_frame_rate(SAMPLING_FREQ)
    audio_sf = s.frame_rate
    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                             :,
                             int(SAMPLING_FREQ * input_start_time): int(
                                 SAMPLING_FREQ * (input_end_time)
                             ),
                             ]
            print(waveform_split.shape)
            if (waveform_split.shape[1] != 0):
                model_outs, _ = model(waveform_split)
                emissions_ = model_outs[0]
                emission_start_frame = time_to_frame(segment_start_time)
                emission_end_frame = time_to_frame(segment_end_time)
                offset = time_to_frame(input_start_time)

                emissions_ = emissions_[
                            emission_start_frame - offset: emission_end_frame - offset, :
                            ]
                emissions_arr.append(emissions_)
                i += EMISSION_INTERVAL
                print(f'i:{i}, total_duration:{total_duration}')
            else:
                break

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]

    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    input_lengths = torch.tensor(emissions.shape[0])
    target_lengths = torch.tensor(targets.shape[0])

    path, _ = F.forced_align(
        emissions[None, :], targets[None, :], input_lengths.ravel(), target_lengths.ravel(), blank=blank
    )
    path = path.to("cpu").tolist()
    segments = merge_repeats(path[0], {v: k for k, v in dictionary.items()})

    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)

    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/manifest.json", "w") as f:
        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end

            output_file = f"{outdir}/segment{i}.wav"

            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000

            extract = s[audio_start_sec * 1000:audio_end_sec * 1000]
            extract.export(output_file, format="wav")

            # tfm = sox.Transformer()
            # tfm.trim(audio_start_sec , audio_end_sec)
            # tfm.build_file(audio_filepath, output_file)

            sample = {
                "audio_start_sec": audio_start_sec,
                "audio_filepath": str(output_file),
                "duration": audio_end_sec - audio_start_sec,
                "text": t,
                "normalized_text": norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            f.write(json.dumps(sample) + "\n")
#%%
force('<folder_name>', '<sentence_level_aligned_text.txt>', '<wav_audio_file>')
