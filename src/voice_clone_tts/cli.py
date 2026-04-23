from __future__ import annotations

import argparse
import os
from pathlib import Path

from .audio import prepare_reference_voice
from .xtts import SynthesisSettings, XttsVoiceCloner, validate_xtts_model_dir


def env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


def read_text_arg(args: argparse.Namespace) -> str:
    if args.text_file:
        return args.text_file.read_text(encoding="utf-8")
    if args.text:
        return args.text
    raise SystemExit("Pass either --text or --text-file.")


def cmd_check_model(args: argparse.Namespace) -> None:
    missing = validate_xtts_model_dir(args.model_dir)
    if missing:
        raise SystemExit(f"Missing files in {args.model_dir}: {', '.join(missing)}")
    print(f"ok: {args.model_dir}")


def cmd_prepare_reference(args: argparse.Namespace) -> None:
    out = prepare_reference_voice(
        args.input,
        args.output,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
    )
    print(out)


def cmd_synthesize(args: argparse.Namespace) -> None:
    model_dir = args.model_dir or env_path("VOICE_CLONE_MODEL_DIR")
    cache_dir = args.cache_dir or env_path("VOICE_CLONE_CACHE_DIR")
    reference = args.reference or env_path("VOICE_CLONE_REFERENCE")
    if reference is None:
        raise SystemExit("Pass --reference or set VOICE_CLONE_REFERENCE.")

    text = read_text_arg(args)
    settings = SynthesisSettings(
        language=args.language,
        chunk_limit=args.chunk_limit,
        paragraph_gap_ms=args.paragraph_gap_ms,
        sentence_gap_ms=args.sentence_gap_ms,
        speed=args.speed,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    cloner = XttsVoiceCloner(model_dir=model_dir, cache_dir=cache_dir, use_gpu=args.gpu)
    out = cloner.synthesize_to_file(
        text=text,
        reference_wav=reference,
        output_path=args.output,
        settings=settings,
    )
    print(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voice-clone-tts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-model", help="Validate a local XTTS model directory.")
    check.add_argument("--model-dir", type=Path, required=True)
    check.set_defaults(func=cmd_check_model)

    prepare = subparsers.add_parser("prepare-reference", help="Convert a recording into a clean reference WAV.")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--start-sec", type=float, default=0.0)
    prepare.add_argument("--duration-sec", type=float, default=45.0)
    prepare.set_defaults(func=cmd_prepare_reference)

    synth = subparsers.add_parser("synthesize", help="Synthesize text in the target voice.")
    synth.add_argument("--text", default="")
    synth.add_argument("--text-file", type=Path)
    synth.add_argument("--reference", type=Path)
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--model-dir", type=Path)
    synth.add_argument("--cache-dir", type=Path)
    synth.add_argument("--language", default="ru")
    synth.add_argument("--chunk-limit", type=int, default=240)
    synth.add_argument("--paragraph-gap-ms", type=int, default=850)
    synth.add_argument("--sentence-gap-ms", type=int, default=360)
    synth.add_argument("--speed", type=float, default=0.93)
    synth.add_argument("--temperature", type=float, default=0.72)
    synth.add_argument("--repetition-penalty", type=float, default=6.5)
    synth.add_argument("--top-k", type=int, default=40)
    synth.add_argument("--top-p", type=float, default=0.9)
    synth.add_argument("--gpu", action="store_true")
    synth.set_defaults(func=cmd_synthesize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
