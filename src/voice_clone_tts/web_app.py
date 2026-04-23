from __future__ import annotations

import argparse
import os
import traceback
from datetime import datetime
from pathlib import Path

try:
    import gradio as gr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install gradio to run the web app: `pip install gradio`.") from exc

from .finetuned import FineTunedXttsSynthesizer
from .training import TrainingSegment, train_from_segments
from .xtts import SynthesisSettings


LANGUAGES = ["ru", "en", "es", "fr", "de", "it", "pt", "pl", "tr", "nl", "cs", "ar", "zh", "hu", "ko", "ja"]


def read_transcript(transcript_text: str, transcript_file: str | None) -> str:
    if transcript_file:
        return Path(transcript_file).read_text(encoding="utf-8").strip()
    return transcript_text.strip()


def read_generation_text(text: str, text_file: str | None) -> str:
    if text_file:
        return Path(text_file).read_text(encoding="utf-8").strip()
    return text.strip()


def safe_run_dir(base_dir: str, speaker: str) -> Path:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in speaker).strip("_") or "speaker"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir).expanduser() / f"{cleaned}_{stamp}"


def train_ui(
    audio_1: str | None,
    transcript_text_1: str,
    transcript_file_1: str | None,
    audio_2: str | None,
    transcript_text_2: str,
    transcript_file_2: str | None,
    audio_3: str | None,
    transcript_text_3: str,
    transcript_file_3: str | None,
    audio_4: str | None,
    transcript_text_4: str,
    transcript_file_4: str | None,
    audio_5: str | None,
    transcript_text_5: str,
    transcript_file_5: str | None,
    language: str,
    speaker_name: str,
    output_root: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_clip_sec: int,
    progress=gr.Progress(track_tqdm=True),
):
    del progress
    raw_segments = [
        (audio_1, transcript_text_1, transcript_file_1),
        (audio_2, transcript_text_2, transcript_file_2),
        (audio_3, transcript_text_3, transcript_file_3),
        (audio_4, transcript_text_4, transcript_file_4),
        (audio_5, transcript_text_5, transcript_file_5),
    ]
    try:
        segments: list[TrainingSegment] = []
        missing_transcripts: list[str] = []
        for idx, (audio_file, transcript_text, transcript_file) in enumerate(raw_segments, start=1):
            if not audio_file:
                continue
            transcript = read_transcript(transcript_text, transcript_file)
            if not transcript:
                missing_transcripts.append(str(idx))
                continue
            segments.append(TrainingSegment(audio_path=Path(audio_file), transcript=transcript))

        if not segments:
            return "Upload at least one audio segment and its exact transcript.", "", "", "", "", ""
        if missing_transcripts:
            joined = ", ".join(missing_transcripts)
            return f"Missing exact transcript for segment(s): {joined}.", "", "", "", "", ""

        run_dir = safe_run_dir(output_root, speaker_name)
        artifacts = train_from_segments(
            segments=segments,
            output_dir=run_dir,
            language=language,
            speaker=speaker_name,
            num_epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_clip_sec=max_clip_sec,
        )
        status = "\n".join(
            [
                "Training completed.",
                f"Training segments: {len(segments)}",
                "",
                "Dataset stats:",
                artifacts.stats,
                "",
                f"run_dir: {artifacts.run_dir}",
                f"checkpoint: {artifacts.checkpoint}",
                f"speaker_wav: {artifacts.speaker_wav}",
            ]
        )
        return (
            status,
            str(artifacts.checkpoint),
            str(artifacts.config),
            str(artifacts.vocab),
            str(artifacts.speaker_wav),
            str(artifacts.run_dir),
        )
    except Exception:
        return traceback.format_exc(), "", "", "", "", ""


def synthesize_ui(
    checkpoint: str,
    config: str,
    vocab: str,
    speaker_wav: str,
    text: str,
    text_file: str | None,
    language: str,
    output_dir: str,
    use_gpu: bool,
    speed: float,
    temperature: float,
):
    try:
        final_text = read_generation_text(text, text_file)
        if not final_text:
            return "Upload or paste text to synthesize.", None
        out_dir = Path(output_dir).expanduser()
        out_path = out_dir / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        synth = FineTunedXttsSynthesizer(
            checkpoint=Path(checkpoint),
            config=Path(config),
            vocab=Path(vocab),
            use_gpu=use_gpu,
        )
        result = synth.synthesize_to_file(
            text=final_text,
            speaker_wav=Path(speaker_wav),
            output_path=out_path,
            settings=SynthesisSettings(language=language, speed=speed, temperature=temperature),
        )
        return f"Generated: {result}", str(result)
    except Exception:
        return traceback.format_exc(), None


def build_app(default_output: Path) -> gr.Blocks:
    css = """
    .hero {background: linear-gradient(135deg, #101820, #263b4a 45%, #8a5a44); padding: 26px; border-radius: 18px; color: white;}
    .hero h1 {margin: 0 0 8px; font-size: 34px;}
    .hero p {font-size: 16px; opacity: 0.92;}
    """
    with gr.Blocks(title="Voice Clone Trainer", css=css) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Voice Clone Trainer</h1>
              <p>Upload a consented voice recording, paste the exact transcript, fine-tune XTTS, then narrate any new text with the trained voice.</p>
            </div>
            """
        )
        gr.Markdown(
            """
            **Critical quality rule:** every transcript must match its audio segment word-for-word. If the text and audio differ, training quality drops and the model can learn wrong pronunciation or unstable intonation.

            Upload 1-5 clean reference segments. More exact segments usually produce better speaker similarity than one long recording.
            """
        )

        with gr.Tab("1. Train Voice"):
            with gr.Row():
                with gr.Column(scale=1):
                    segment_inputs = []
                    for idx in range(1, 6):
                        with gr.Accordion(f"Reference segment {idx}", open=idx == 1):
                            audio = gr.Audio(
                                label=f"Insert audio file {idx}",
                                type="filepath",
                                sources=["upload"],
                            )
                            transcript_file = gr.File(
                                label=f"Optional transcript .txt {idx}",
                                file_types=[".txt"],
                                type="filepath",
                            )
                            transcript_text = gr.Textbox(
                                label=f"Or paste exact transcript {idx}",
                                lines=4,
                            )
                            segment_inputs.extend([audio, transcript_text, transcript_file])
                    language = gr.Dropdown(label="Language", choices=LANGUAGES, value="ru")
                with gr.Column(scale=1):
                    speaker_name = gr.Textbox(label="Speaker name", value="trained_voice")
                    output_root = gr.Textbox(label="Training output folder", value=str(default_output / "runs"))
                    epochs = gr.Slider(label="Epochs", minimum=1, maximum=50, step=1, value=6)
                    batch_size = gr.Slider(label="Batch size", minimum=1, maximum=16, step=1, value=2)
                    grad_accum = gr.Slider(label="Gradient accumulation", minimum=1, maximum=16, step=1, value=2)
                    max_clip_sec = gr.Slider(label="Max clip length, sec", minimum=4, maximum=30, step=1, value=12)
                    train_button = gr.Button("Train voice", variant="primary")

            train_status = gr.Textbox(label="Training status", lines=14)
            with gr.Accordion("Model artifacts", open=True):
                checkpoint = gr.Textbox(label="Fine-tuned checkpoint")
                config = gr.Textbox(label="XTTS config")
                vocab = gr.Textbox(label="XTTS vocab")
                speaker_wav = gr.Textbox(label="Speaker reference WAV")
                run_dir = gr.Textbox(label="Run directory")

            train_button.click(
                train_ui,
                inputs=[
                    *segment_inputs,
                    language,
                    speaker_name,
                    output_root,
                    epochs,
                    batch_size,
                    grad_accum,
                    max_clip_sec,
                ],
                outputs=[train_status, checkpoint, config, vocab, speaker_wav, run_dir],
            )

        with gr.Tab("2. Generate New Audio"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_file = gr.File(label="Insert text file for narration", file_types=[".txt"], type="filepath")
                    text = gr.Textbox(label="Or paste text to narrate", lines=10)
                    gen_language = gr.Dropdown(label="Language", choices=LANGUAGES, value="ru")
                    gen_output_dir = gr.Textbox(label="Audio output folder", value=str(default_output / "output"))
                with gr.Column(scale=1):
                    use_gpu = gr.Checkbox(label="Use GPU if available", value=False)
                    speed = gr.Slider(label="Speed", minimum=0.7, maximum=1.2, step=0.01, value=0.93)
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.2, step=0.01, value=0.72)
                    gen_button = gr.Button("Generate audio file", variant="primary")

            gen_status = gr.Textbox(label="Generation status", lines=8)
            gen_audio = gr.Audio(label="Generated audio", type="filepath")

            gen_button.click(
                synthesize_ui,
                inputs=[
                    checkpoint,
                    config,
                    vocab,
                    speaker_wav,
                    text,
                    text_file,
                    gen_language,
                    gen_output_dir,
                    use_gpu,
                    speed,
                    temperature,
                ],
                outputs=[gen_status, gen_audio],
            )

    return demo


def main(default_output_dir: Path | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir or Path("workspace"))
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(args.output_dir)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
