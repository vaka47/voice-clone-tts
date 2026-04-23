from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .audio import concat_wavs
from .text import split_text


XTTS_REQUIRED_FILES = (
    "config.json",
    "model.pth",
    "speakers_xtts.pth",
    "vocab.json",
)


@dataclass(frozen=True)
class SynthesisSettings:
    language: str = "ru"
    chunk_limit: int = 240
    paragraph_gap_ms: int = 850
    sentence_gap_ms: int = 360
    speed: float = 0.93
    temperature: float = 0.72
    repetition_penalty: float = 6.5
    top_k: int = 40
    top_p: float = 0.9


def configure_cache(cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TTS_HOME", str(cache_dir))
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hf" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "hf" / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mpl"))
    os.environ.setdefault("COQUI_TOS_AGREED", "1")


def validate_xtts_model_dir(model_dir: Path) -> list[str]:
    return [name for name in XTTS_REQUIRED_FILES if not (model_dir / name).exists()]


class XttsVoiceCloner:
    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        cache_dir: Path | None = None,
        use_gpu: bool = False,
    ) -> None:
        configure_cache(cache_dir)
        self.model_dir = model_dir
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.tts = self._load_tts()

    def _load_tts(self):
        try:
            from TTS.api import TTS
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Coqui TTS is not installed. Run `pip install -r requirements.txt`.") from exc

        if self.model_dir is None:
            return TTS(model_name=self.model_name, progress_bar=True, gpu=self.use_gpu)

        missing = validate_xtts_model_dir(self.model_dir)
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"Missing XTTS model files in {self.model_dir}: {joined}")

        from TTS.utils.synthesizer import Synthesizer

        tts = TTS(model_name=None, progress_bar=False, gpu=self.use_gpu)
        tts.model_name = self.model_name
        tts.synthesizer = Synthesizer(model_dir=str(self.model_dir), use_cuda=self.use_gpu)
        return tts

    def synthesize_to_file(
        self,
        *,
        text: str,
        reference_wav: Path,
        output_path: Path,
        settings: SynthesisSettings,
    ) -> Path:
        if not reference_wav.exists():
            raise FileNotFoundError(reference_wav)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = split_text(
            text,
            chunk_limit=settings.chunk_limit,
            paragraph_gap_ms=settings.paragraph_gap_ms,
            sentence_gap_ms=settings.sentence_gap_ms,
        )
        if not chunks:
            raise ValueError("Input text is empty after normalization.")

        with tempfile.TemporaryDirectory(prefix="voice-clone-tts-") as tmp_raw:
            tmp_dir = Path(tmp_raw)
            parts: list[Path] = []
            gaps: list[int] = []

            for idx, (chunk, gap_after_ms) in enumerate(chunks, start=1):
                part_path = tmp_dir / f"part_{idx:04d}.wav"
                wav = self.tts.synthesizer.tts(
                    text=chunk,
                    speaker_wav=str(reference_wav),
                    language_name=settings.language,
                    speed=settings.speed,
                    temperature=settings.temperature,
                    length_penalty=1.0,
                    repetition_penalty=settings.repetition_penalty,
                    top_k=settings.top_k,
                    top_p=settings.top_p,
                    gpt_cond_len=18,
                    gpt_cond_chunk_len=6,
                    max_ref_len=18,
                    enable_text_splitting=True,
                    split_sentences=False,
                )
                self.tts.synthesizer.save_wav(wav=wav, path=str(part_path))
                parts.append(part_path)
                gaps.append(gap_after_ms)

            return concat_wavs(parts, gaps, output_path)
