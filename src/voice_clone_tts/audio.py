from __future__ import annotations

import wave
from pathlib import Path


TARGET_SAMPLE_RATE = 24_000


def prepare_reference_voice(
    input_path: Path,
    output_path: Path,
    *,
    start_sec: float = 0.0,
    duration_sec: float | None = 45.0,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """Convert an audio file into a mono PCM WAV suitable for XTTS speaker_wav."""
    try:
        import librosa
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Install librosa and soundfile to prepare reference audio.") from exc

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    offset = max(0.0, start_sec)
    duration = None if duration_sec is None or duration_sec <= 0 else duration_sec
    audio, sr = librosa.load(str(input_path), sr=sample_rate, mono=True, offset=offset, duration=duration)
    if audio.size == 0:
        raise ValueError(f"No audio loaded from {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, subtype="PCM_16")
    return output_path


def concat_wavs(parts: list[Path], gaps_ms: list[int], output_path: Path) -> Path:
    if not parts:
        raise ValueError("No WAV parts to concatenate.")

    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
        frame_rate = first.getframerate()
        sample_width = first.getsampwidth()
        channels = first.getnchannels()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        for idx, part in enumerate(parts):
            with wave.open(str(part), "rb") as src:
                if src.getframerate() != frame_rate or src.getnchannels() != channels:
                    raise ValueError(f"Incompatible WAV parameters in {part}")
                out.writeframes(src.readframes(src.getnframes()))

            gap_ms = gaps_ms[idx] if idx < len(gaps_ms) else 0
            if gap_ms > 0:
                silence_frames = int(frame_rate * gap_ms / 1000)
                silence = b"\x00" * silence_frames * sample_width * channels
                out.writeframes(silence)

    return output_path
