from __future__ import annotations

import csv
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .audio import TARGET_SAMPLE_RATE
from .dataset import dataset_stats, format_stats
from .text import split_paragraph


@dataclass(frozen=True)
class TrainingArtifacts:
    dataset_dir: Path
    train_csv: Path
    eval_csv: Path
    checkpoint: Path
    config: Path
    vocab: Path
    speaker_wav: Path
    run_dir: Path
    stats: str


def optional_site_packages() -> Path | None:
    raw = os.environ.get("VOICE_CLONE_SITE_PACKAGES")
    return Path(raw).expanduser() if raw else None


def add_optional_site_packages() -> None:
    site_packages = optional_site_packages()
    if site_packages and site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))


def split_transcript(text: str, max_chars: int = 180) -> list[str]:
    paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        chunks.extend(split_paragraph(paragraph, limit=max_chars))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def write_wav(path: Path, audio, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install soundfile to write dataset clips.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def load_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE):
    try:
        import librosa
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install librosa to load and resample training audio.") from exc
    audio, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"No audio loaded from {path}")
    return audio, sr


def build_exact_transcript_dataset(
    *,
    audio_path: Path,
    transcript: str,
    output_dir: Path,
    speaker: str = "speaker",
    language: str = "ru",
    max_clip_sec: float = 12.0,
    max_chars: int = 180,
    eval_ratio: float = 0.15,
) -> tuple[Path, Path, str]:
    """Build Coqui XTTS fine-tuning CSVs from one recording and a matching transcript.

    If the recording is longer than max_clip_sec, it is sliced into transcript chunks
    proportionally by character count. This is a practical portfolio workflow; for
    production quality, use manually segmented clips or forced alignment.
    """
    audio_path = audio_path.expanduser()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    audio, sr = load_audio(audio_path)
    total_samples = len(audio)
    total_seconds = total_samples / sr

    chunks = split_transcript(transcript, max_chars=max_chars)
    if not chunks:
        raise ValueError("Transcript produced no usable chunks.")

    if total_seconds <= max_clip_sec:
        chunks = [transcript]

    weights = [max(1, len(chunk)) for chunk in chunks]
    total_weight = sum(weights)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    cursor = 0
    for idx, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
        if idx == len(chunks):
            end = total_samples
        else:
            end = min(total_samples, cursor + math.floor(total_samples * weight / total_weight))
        if end <= cursor:
            continue

        stem = f"{speaker}_{idx:05d}"
        rel_audio = f"wavs/{stem}.wav"
        write_wav(wavs_dir / f"{stem}.wav", audio[cursor:end], sr)
        rows.append({"audio_file": rel_audio, "text": chunk, "speaker_name": speaker})
        cursor = end

    if len(rows) < 2:
        # Coqui training expects train/eval paths. Duplicate the single row for a
        # minimal smoke run; real fine-tuning should use many clips.
        rows = rows * 2

    split_at = max(1, int(len(rows) * (1 - eval_ratio)))
    if split_at >= len(rows):
        split_at = len(rows) - 1
    train_rows = rows[:split_at]
    eval_rows = rows[split_at:]

    train_csv = output_dir / "metadata_train.csv"
    eval_csv = output_dir / "metadata_eval.csv"
    for path, data in ((train_csv, train_rows), (eval_csv, eval_rows)):
        with path.open("w", encoding="utf-8", newline="") as raw:
            writer = csv.DictWriter(raw, fieldnames=["audio_file", "text", "speaker_name"], delimiter="|")
            writer.writeheader()
            writer.writerows(data)

    # Also write LJSpeech-style metadata.csv for simpler external inspection.
    metadata_lines = []
    for row in rows:
        stem = Path(row["audio_file"]).stem
        metadata_lines.append(f"{stem}|{row['text']}|{row['text']}")
    (output_dir / "metadata.csv").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")

    return train_csv, eval_csv, format_stats(dataset_stats(output_dir))


def train_xtts_gpt(
    *,
    language: str,
    train_csv: Path,
    eval_csv: Path,
    output_dir: Path,
    num_epochs: int = 6,
    batch_size: int = 2,
    grad_accum: int = 2,
    max_audio_length_sec: int = 12,
) -> tuple[Path, Path, Path, Path, Path]:
    """Run Coqui XTTS GPT encoder fine-tuning and return model artifacts."""
    add_optional_site_packages()
    try:
        from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not import Coqui XTTS fine-tuning utilities. "
            "Install TTS with demo utilities or set VOICE_CLONE_SITE_PACKAGES."
        ) from exc

    config_path, _original_checkpoint, vocab_path, run_dir, speaker_wav = train_gpt(
        language,
        num_epochs,
        batch_size,
        grad_accum,
        str(train_csv),
        str(eval_csv),
        output_path=str(output_dir),
        max_audio_length=int(max_audio_length_sec * 22050),
    )

    run_dir_path = Path(run_dir)
    config = Path(config_path)
    vocab = Path(vocab_path)
    checkpoint = run_dir_path / "best_model.pth"
    if config.exists():
        shutil.copy2(config, run_dir_path / config.name)
    if vocab.exists():
        shutil.copy2(vocab, run_dir_path / vocab.name)

    return checkpoint, config, vocab, run_dir_path, Path(speaker_wav)


def train_from_single_recording(
    *,
    audio_path: Path,
    transcript: str,
    output_dir: Path,
    language: str = "ru",
    speaker: str = "speaker",
    num_epochs: int = 6,
    batch_size: int = 2,
    grad_accum: int = 2,
    max_clip_sec: int = 12,
) -> TrainingArtifacts:
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv, stats = build_exact_transcript_dataset(
        audio_path=audio_path,
        transcript=transcript,
        output_dir=dataset_dir,
        speaker=speaker,
        language=language,
        max_clip_sec=max_clip_sec,
    )
    checkpoint, config, vocab, run_dir, speaker_wav = train_xtts_gpt(
        language=language,
        train_csv=train_csv,
        eval_csv=eval_csv,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_audio_length_sec=max_clip_sec,
    )
    return TrainingArtifacts(
        dataset_dir=dataset_dir,
        train_csv=train_csv,
        eval_csv=eval_csv,
        checkpoint=checkpoint,
        config=config,
        vocab=vocab,
        speaker_wav=speaker_wav,
        run_dir=run_dir,
        stats=stats,
    )
