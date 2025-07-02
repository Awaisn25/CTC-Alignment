# CTC Forced Alignment Script

This script performs forced alignment of audio and transcript using a CTC-based model with uroman romanization and text normalization. It segments the audio according to the transcript and outputs aligned audio segments and a manifest file.

## Features

- Text normalization and romanization (using uroman)
- Forced alignment using a pretrained CTC model
- Audio segmentation and export of aligned segments
- Manifest file generation with segment metadata

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [pydub](https://github.com/jiaaro/pydub)
- [soundfile](https://pysoundfile.readthedocs.io/)
- [uroman](https://github.com/isi-nlp/uroman) (Perl version, required for romanization)
- sox (optional, for alternative audio processing)
- Other dependencies: `requests`, `bs4`, `unidecode` (for some normalization)

## Usage

1. **Prepare your files:**
   - A plain text file with one transcript line per audio segment (e.g., `sentence_level_aligned_text.txt`)
   - The corresponding audio file (e.g., `wav_audio_file`)
   - A folder to store the output (e.g., `<folder_name>`)

2. **Run the script:**

   You can execute the alignment by running the following command in a Python environment:

   ```python
   from scrap_ import force
   force('<folder_name>', '<sentence_level_aligned_text.txt>', '<wav_audio_file>')
   ```

   Or, if running interactively (as in the script's last line):

   ```python
   force('<folder_name>', '<sentence_level_aligned_text.txt>', '<wav_audio_file>')
   ```

   Replace `<folder_name>`, `<sentence_level_aligned_text.txt>`, and `<wav_audio_file>` with your actual output directory, transcript file, and audio file paths.

3. **Output:**
   - Segmented audio files will be saved in `<folder_name>/segment{i}.wav`
   - A manifest file with alignment metadata will be saved as `<folder_name>/manifest.json`

## Notes

- The script expects the uroman Perl scripts and model/dictionary files to be available in the specified paths.
- The language code for romanization is set to `'pus'` (Pashto) by default; modify as needed for your data.
- The script will download the CTC model and dictionary if not already present.

## Example

```python
force('Aligned', 'FullAudio29.txt', 'aud.mp3')
```

This will align `aud.mp3` using the transcripts in `FullAudio29.txt` and output the results in the
