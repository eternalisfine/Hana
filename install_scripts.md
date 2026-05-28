# Installation

## Linux (Ubuntu / Debian)
```bash
sudo apt install ffmpeg python3-pip && pip install PySide6 faster-whisper sounddevice soundfile numpy torch torchaudio silero-vad requests ginza ja-ginza spacy
```

## Arch Linux
```bash
sudo pacman -S ffmpeg python-pip && pip install PySide6 faster-whisper sounddevice soundfile numpy torch torchaudio silero-vad requests ginza ja-ginza spacy
```

## macOS
```bash
brew install ffmpeg && pip install PySide6 faster-whisper sounddevice soundfile numpy torch torchaudio silero-vad requests ginza ja-ginza spacy
```

## Windows (PowerShell)
> Install ffmpeg first: https://ffmpeg.org/download.html
```powershell
pip install PySide6 faster-whisper sounddevice soundfile numpy torch torchaudio silero-vad requests ginza ja-ginza spacy
```

If `torch` fails on Windows, run these two steps instead:
```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install PySide6 faster-whisper sounddevice soundfile numpy silero-vad requests ginza ja-ginza spacy
```