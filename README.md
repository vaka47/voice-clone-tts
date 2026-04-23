# Voice Clone TTS Pipeline

Clean portfolio project for local voice cloning and text-to-speech generation.

The pipeline accepts up to five consented reference recordings with exact transcripts, prepares a fine-tuning dataset, trains an XTTS speaker adaptation, and synthesizes arbitrary text in the trained voice.

## What This Project Demonstrates

- Local-first voice cloning workflow with XTTS.
- Reference voice preprocessing: mono conversion, resampling, trimming.
- Chunked long-text synthesis with configurable pauses.
- Fine-tuning dataset preparation from audio+transcript manifests.
- Web app for training a voice from up to five aligned reference segments.
- Fine-tuned checkpoint inference for arbitrary text files.
- Reproducible CLI for turning any text file into narrated audio.
- Safe project structure: no model weights, private voice samples, or generated audio committed to git.

## Important Note About Training

XTTS v2 supports two practical modes:

- **Zero-shot voice cloning:** the pretrained model uses a reference recording at inference time and does not change weights.
- **Fine-tuning:** the GPT encoder is adapted on `audio + exact transcript` pairs, producing a speaker-specific checkpoint.

This repository includes both the zero-shot CLI path and the fine-tuning web app path.

## Repository Structure

```text
src/voice_clone_tts/
  cli.py          # command-line interface
  audio.py        # reference voice preparation and WAV concatenation
  dataset.py      # fine-tuning dataset preparation and stats
  training.py     # XTTS fine-tuning workflow
  finetuned.py    # inference from a fine-tuned checkpoint
  web_app.py      # Gradio UI for training and generation
  text.py         # text normalization and chunking
  xtts.py         # XTTS model loading and synthesis
samples/
  README.md       # local-only voice samples
examples/
  sample_text.txt
docs/
  ETHICS.md
  ARCHITECTURE.md
  FINE_TUNING.md
  PORTABLE_APP.md
```

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Web App

Run the portfolio UI:

```bash
python -m voice_clone_tts.web_app
```

Open the local URL shown in the terminal. The app exposes the main workflow:

1. upload 1-5 consented speech recordings;
2. paste or upload the exact transcript for each recording;
3. click **Train voice**;
4. upload a new text file;
5. click **Generate audio file**.

The transcript must be identical to the spoken audio. If words are missing, added, or reordered, the fine-tuned model can learn unstable pronunciation and intonation.

## Downloadable App Builds

The project includes PyInstaller build scripts and a GitHub Actions workflow for portable macOS and Windows builds:

- `VoiceCloneTrainer-macOS.zip`
- `VoiceCloneTrainer-Windows.zip`

See [docs/PORTABLE_APP.md](docs/PORTABLE_APP.md).

Prepare a reference voice:

```bash
python -m voice_clone_tts.cli prepare-reference \
  --input /path/to/consented_voice_recording.mp3 \
  --output samples/reference_voice.wav \
  --start-sec 30 \
  --duration-sec 45
```

Synthesize text:

```bash
python -m voice_clone_tts.cli synthesize \
  --text-file examples/sample_text.txt \
  --reference samples/reference_voice.wav \
  --output output/demo.wav \
  --model-dir /path/to/tts_models--multilingual--multi-dataset--xtts_v2 \
  --language ru
```

You can also configure paths with environment variables:

```bash
cp .env.example .env
```

## Local Model

This project expects XTTS v2 model files locally when using `--model-dir`.

Required files:

- `config.json`
- `model.pth`
- `speakers_xtts.pth`
- `vocab.json`

Example path from a local cache:

```text
/Volumes/Untitled/kron_voiceclone/cache/tts/tts_models--multilingual--multi-dataset--xtts_v2
```

Do not commit model files to git.

## CLI Commands

Check local XTTS files:

```bash
python -m voice_clone_tts.cli check-model --model-dir /path/to/xtts_v2
```

Prepare a fine-tuning dataset from audio+transcript pairs:

```bash
python -m voice_clone_tts.cli build-dataset \
  --manifest examples/manifest.example.csv \
  --output-dir data/processed/demo_speaker
```

Print dataset statistics:

```bash
python -m voice_clone_tts.cli dataset-stats \
  --dataset-dir data/processed/demo_speaker
```

Prepare a reference:

```bash
python -m voice_clone_tts.cli prepare-reference \
  --input raw_voice.mp3 \
  --output samples/reference_voice.wav
```

Render from inline text:

```bash
python -m voice_clone_tts.cli synthesize \
  --text "Привет. Это тест синтеза речи." \
  --reference samples/reference_voice.wav \
  --output output/test.wav \
  --model-dir /path/to/xtts_v2
```

Render from a text file:

```bash
python -m voice_clone_tts.cli synthesize \
  --text-file examples/sample_text.txt \
  --reference samples/reference_voice.wav \
  --output output/sample.wav \
  --model-dir /path/to/xtts_v2
```

## Responsible Use

Only clone voices you own or have explicit permission to use. Do not publish reference samples or generated voice outputs without consent.

See [docs/ETHICS.md](docs/ETHICS.md).

## Fine-Tuning

See [docs/FINE_TUNING.md](docs/FINE_TUNING.md) for the difference between zero-shot voice cloning and actual weight fine-tuning.

## Portfolio Positioning

Recommended GitHub description:

> Local XTTS voice-cloning pipeline for preparing reference speech and synthesizing arbitrary text with a consented target voice.

This project is stronger as a portfolio item than the book-specific version because it is reusable, focused, and easier for reviewers to understand.
