$ErrorActionPreference = "Stop"

Set-Location (Split-Path -Parent $PSScriptRoot)

python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt

python -m PyInstaller `
  --clean `
  --noconfirm `
  --name VoiceCloneTrainer `
  --windowed `
  --paths src `
  --collect-all gradio `
  --collect-all TTS `
  --collect-all trainer `
  --collect-all coqpit `
  launcher.py

New-Item -ItemType Directory -Force -Path release | Out-Null
Compress-Archive -Path dist/VoiceCloneTrainer -DestinationPath release/VoiceCloneTrainer-Windows.zip -Force

Write-Output "release/VoiceCloneTrainer-Windows.zip"
