from __future__ import annotations

import tempfile
from pathlib import Path

from .audio import concat_wavs
from .text import split_text
from .training import add_optional_site_packages
from .xtts import SynthesisSettings


class FineTunedXttsSynthesizer:
    def __init__(self, *, checkpoint: Path, config: Path, vocab: Path, use_gpu: bool = False) -> None:
        add_optional_site_packages()
        try:
            import torch
            import torchaudio
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Coqui TTS fine-tuned inference dependencies are not installed.") from exc

        self.torch = torch
        self.torchaudio = torchaudio
        self.use_gpu = use_gpu and torch.cuda.is_available()

        xtts_config = XttsConfig()
        xtts_config.load_json(str(config))
        self.model = Xtts.init_from_config(xtts_config)
        self.model.load_checkpoint(xtts_config, checkpoint_path=str(checkpoint), vocab_path=str(vocab), use_deepspeed=False)
        if self.use_gpu:
            self.model.cuda()

    def synthesize_to_file(
        self,
        *,
        text: str,
        speaker_wav: Path,
        output_path: Path,
        settings: SynthesisSettings,
    ) -> Path:
        if not speaker_wav.exists():
            raise FileNotFoundError(speaker_wav)

        chunks = split_text(
            text,
            chunk_limit=settings.chunk_limit,
            paragraph_gap_ms=settings.paragraph_gap_ms,
            sentence_gap_ms=settings.sentence_gap_ms,
        )
        if not chunks:
            raise ValueError("Input text is empty after normalization.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=str(speaker_wav),
            gpt_cond_len=self.model.config.gpt_cond_len,
            max_ref_length=self.model.config.max_ref_len,
            sound_norm_refs=self.model.config.sound_norm_refs,
        )

        with tempfile.TemporaryDirectory(prefix="fine-tuned-xtts-") as tmp_raw:
            tmp_dir = Path(tmp_raw)
            parts: list[Path] = []
            gaps: list[int] = []
            for idx, (chunk, gap_after_ms) in enumerate(chunks, start=1):
                out = self.model.inference(
                    text=chunk,
                    language=settings.language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=settings.temperature,
                    length_penalty=1.0,
                    repetition_penalty=settings.repetition_penalty,
                    top_k=settings.top_k,
                    top_p=settings.top_p,
                )
                part_path = tmp_dir / f"part_{idx:04d}.wav"
                wav = self.torch.tensor(out["wav"]).unsqueeze(0)
                self.torchaudio.save(str(part_path), wav, 24000)
                parts.append(part_path)
                gaps.append(gap_after_ms)

            return concat_wavs(parts, gaps, output_path)
