# Voice Clone TTS Pipeline

Clean portfolio project for local voice cloning and text-to-speech generation.

The pipeline accepts a reference speech recording, prepares a clean voice sample, loads a local XTTS model, and synthesizes arbitrary text in the target voice.

## What This Project Demonstrates

- Local-first voice cloning workflow with XTTS.
- Reference voice preprocessing: mono conversion, resampling, trimming.
- Chunked long-text synthesis with configurable pauses.
- Reproducible CLI for turning any text file into narrated audio.
- Safe project structure: no model weights, private voice samples, or generated audio committed to git.

## Important Note About "Training"

XTTS v2 usually works through **zero-shot voice cloning**: it does not fine-tune model weights for every new speaker. Instead, the model conditions generation on a short reference recording.

In this repository, "create a voice" means:

1. provide a consented reference voice sample;
2. prepare it into a clean 24 kHz mono WAV;
3. use it as `speaker_wav` during synthesis.

Full speaker fine-tuning is intentionally not included in this portfolio version.

## Repository Structure

```text
src/voice_clone_tts/
  cli.py          # command-line interface
  audio.py        # reference voice preparation and WAV concatenation
  text.py         # text normalization and chunking
  xtts.py         # XTTS model loading and synthesis
samples/
  README.md       # local-only voice samples
examples/
  sample_text.txt
docs/
  ETHICS.md
  ARCHITECTURE.md
```

## Quick Start

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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

## Portfolio Positioning

Recommended GitHub description:

> Local XTTS voice-cloning pipeline for preparing reference speech and synthesizing arbitrary text with a consented target voice.

This project is stronger as a portfolio item than the book-specific version because it is reusable, focused, and easier for reviewers to understand.
