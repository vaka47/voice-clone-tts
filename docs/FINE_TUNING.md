# Fine-Tuning Guide

## What Fine-Tuning Weights Means

Zero-shot voice cloning uses a pretrained model and a short reference sample at inference time. The model weights stay unchanged.

Fine-tuning changes part of the model weights on a speaker-specific dataset. In practice, the model sees many examples like:

```text
audio clip -> exact transcript
```

and adjusts itself to reproduce that speaker, recording style, language, and pronunciation more consistently.

## Why It Can Be Stronger

Fine-tuning can improve:

- speaker similarity;
- pronunciation stability;
- emotional consistency;
- long-form narration stability;
- performance on a specific language or accent.

It can also make results worse if the dataset is noisy, badly transcribed, too small, or overfitted.

## Dataset Requirement

For credible fine-tuning, a single untranscribed MP3 is not enough. A practical dataset should contain:

- 30-120+ minutes of clean speech for experiments;
- 2-10+ hours for stronger results;
- accurate transcripts for every clip;
- consistent recording quality;
- short clips, ideally 3-15 seconds each;
- no music, reverb, overlapping speech, or background noise.

## Portfolio-Friendly Pipeline

This repository supports the first production-relevant fine-tuning step: dataset preparation.

Input manifest:

```csv
audio_path,text,speaker
/path/to/clip_001.wav,Привет. Это первая фраза.,speaker_1
/path/to/clip_002.wav,Это вторая фраза для обучения.,speaker_1
```

Build LJSpeech-style data:

```bash
python -m voice_clone_tts.cli build-dataset \
  --manifest data/manifest.csv \
  --output-dir data/processed/speaker_1
```

Check dataset stats:

```bash
python -m voice_clone_tts.cli dataset-stats \
  --dataset-dir data/processed/speaker_1
```

Expected output:

```text
files: 120
duration: 3840.0s / 64.0min / 1.07h
min_clip: 2.40s
max_clip: 14.80s
avg_clip: 8.30s
total_chars: 42100
```

## XTTS Fine-Tuning Path

According to Coqui XTTS documentation, XTTS v2 supports fine-tuning of the GPT encoder. A practical public project should present this as an optional training stage:

1. collect consented speech;
2. split audio into clean clips;
3. transcribe clips accurately;
4. build LJSpeech-style dataset;
5. run XTTS fine-tuning using Coqui training tools or demo recipes;
6. compare zero-shot vs fine-tuned output;
7. publish metrics and short consented samples only.

## What Would Make This a "Wow" Portfolio Project

The strongest version is not just "voice cloning works". The strongest version is an MLOps-style pipeline:

- dataset builder from manifest;
- dataset stats and validation;
- zero-shot inference baseline;
- optional fine-tuning stage;
- before/after comparison;
- model card;
- clear ethics policy;
- reproducible CLI;
- no private data in git.

## Recommended Public Demo

Use your own voice:

1. record 30-60 minutes of clean speech;
2. transcribe and split it;
3. fine-tune XTTS GPT encoder;
4. publish a short demo comparing:
   - base XTTS zero-shot;
   - fine-tuned XTTS;
   - same text, same reference.

This is a much stronger hiring signal than uploading a model that clones an unknown third-party voice.
