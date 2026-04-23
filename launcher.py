from pathlib import Path

from voice_clone_tts.web_app import main


def default_workspace() -> Path:
    return Path.home() / "VoiceCloneTrainer"


if __name__ == "__main__":
    main(default_output_dir=default_workspace())
