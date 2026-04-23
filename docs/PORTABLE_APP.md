# Portable App Builds

This project can be packaged into downloadable macOS and Windows archives with PyInstaller.

The portable build is distributed as an extracted app folder, not a single self-extracting binary. This is more reliable for PyTorch and XTTS dependencies than a `--onefile` bundle.

## User Flow

1. Download the archive for your OS.
2. Unzip it.
3. Open `VoiceCloneTrainer`.
4. The app starts a local interface and opens it in the default browser.
5. Upload 1-5 voice reference segments.
6. Paste or upload the exact transcript for each segment.
7. Click **Train voice**.
8. Upload a new text file.
9. Click **Generate audio file**.
10. Download the generated WAV.

## Important Limitation

The portable app bundles Python code and dependencies, but it does not commit private voice files, generated audio, or model checkpoints to GitHub.

Depending on the environment, XTTS model files may be downloaded or cached on first training. This is expected: voice models and PyTorch dependencies are too large to treat as normal source-code assets.

## Build Locally On macOS

```bash
bash scripts/build_macos.sh
```

Output:

```text
release/VoiceCloneTrainer-macOS.zip
```

## Build Locally On Windows

```powershell
scripts/build_windows.ps1
```

Output:

```text
release/VoiceCloneTrainer-Windows.zip
```

## GitHub Artifacts

The workflow `.github/workflows/build-portable.yml` builds both artifacts:

- `VoiceCloneTrainer-macOS.zip`
- `VoiceCloneTrainer-Windows.zip`

Run it manually from GitHub Actions or let it run on pushes to the portfolio branch.
