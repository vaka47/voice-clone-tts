#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-build.txt

python3 -m PyInstaller \
  --clean \
  --noconfirm \
  --name VoiceCloneTrainer \
  --onefile \
  --windowed \
  --paths src \
  --collect-all gradio \
  --collect-all TTS \
  --collect-all trainer \
  --collect-all coqpit \
  launcher.py

mkdir -p release
if [[ -d "dist/VoiceCloneTrainer.app" ]]; then
  ditto -c -k --sequesterRsrc --keepParent "dist/VoiceCloneTrainer.app" "release/VoiceCloneTrainer-macOS.zip"
else
  ditto -c -k "dist/VoiceCloneTrainer" "release/VoiceCloneTrainer-macOS.zip"
fi

echo "release/VoiceCloneTrainer-macOS.zip"
