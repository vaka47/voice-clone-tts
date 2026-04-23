from __future__ import annotations

import csv
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path

from .audio import TARGET_SAMPLE_RATE, prepare_reference_voice


@dataclass(frozen=True)
class DatasetItem:
    audio_path: Path
    text: str
    speaker: str = "speaker"


@dataclass(frozen=True)
class DatasetStats:
    files: int
    total_seconds: float
    min_seconds: float
    max_seconds: float
    avg_seconds: float
    total_chars: int


def read_manifest(path: Path) -> list[DatasetItem]:
    """Read a CSV manifest with columns: audio_path,text[,speaker]."""
    items: list[DatasetItem] = []
    with path.open("r", encoding="utf-8", newline="") as raw:
        reader = csv.DictReader(raw)
        required = {"audio_path", "text"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Manifest must contain audio_path and text columns.")

        for row in reader:
            audio_path = Path(row["audio_path"]).expanduser()
            text = (row["text"] or "").strip()
            speaker = (row.get("speaker") or "speaker").strip() or "speaker"
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            if not text:
                raise ValueError(f"Empty transcript for {audio_path}")
            items.append(DatasetItem(audio_path=audio_path, text=text, speaker=speaker))
    return items


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def build_ljspeech_dataset(
    manifest_path: Path,
    output_dir: Path,
    *,
    copy_without_conversion: bool = False,
) -> Path:
    """Create a simple LJSpeech-style dataset from audio+transcript rows."""
    items = read_manifest(manifest_path)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    metadata_lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        stem = f"{item.speaker}_{idx:05d}"
        out_wav = wavs_dir / f"{stem}.wav"
        if copy_without_conversion and item.audio_path.suffix.lower() == ".wav":
            shutil.copy2(item.audio_path, out_wav)
        else:
            prepare_reference_voice(
                item.audio_path,
                out_wav,
                start_sec=0.0,
                duration_sec=None,
                sample_rate=TARGET_SAMPLE_RATE,
            )
        metadata_lines.append(f"{stem}|{item.text}|{item.text}")

    metadata_path = output_dir / "metadata.csv"
    metadata_path.write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    return metadata_path


def dataset_stats(dataset_dir: Path) -> DatasetStats:
    metadata_path = dataset_dir / "metadata.csv"
    wavs_dir = dataset_dir / "wavs"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    if not wavs_dir.exists():
        raise FileNotFoundError(wavs_dir)

    durations: list[float] = []
    total_chars = 0
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        stem, _, text = line.partition("|")
        if "|" in text:
            text = text.split("|", 1)[0]
        wav_path = wavs_dir / f"{stem}.wav"
        durations.append(wav_duration_seconds(wav_path))
        total_chars += len(text)

    total = sum(durations)
    return DatasetStats(
        files=len(durations),
        total_seconds=total,
        min_seconds=min(durations) if durations else 0.0,
        max_seconds=max(durations) if durations else 0.0,
        avg_seconds=total / len(durations) if durations else 0.0,
        total_chars=total_chars,
    )


def format_stats(stats: DatasetStats) -> str:
    minutes = stats.total_seconds / 60
    hours = stats.total_seconds / 3600
    return "\n".join(
        [
            f"files: {stats.files}",
            f"duration: {stats.total_seconds:.1f}s / {minutes:.1f}min / {hours:.2f}h",
            f"min_clip: {stats.min_seconds:.2f}s",
            f"max_clip: {stats.max_seconds:.2f}s",
            f"avg_clip: {stats.avg_seconds:.2f}s",
            f"total_chars: {stats.total_chars}",
        ]
    )
