# はな / Hana — Japanese Conversation Tutor

A fully local, open-source Japanese speaking practice app.  
No internet required. No API costs. Everything runs on your machine.

---

## What it does

- Always listening — no button to press, just start talking
- Automatically stops when you pause and sends your speech to be transcribed
- Responds in natural Japanese, scaled to your level
- Speaks back using VOICEVOX (natural Japanese TTS)
- Remembers your conversation history, mistakes, and speaking style across sessions
- Flags potentially uncertain Japanese with a ⚠ warning
- Interrupts itself immediately if you start speaking

---

## Tech stack

| Component        | Tool                        |
|------------------|-----------------------------|
| Desktop UI       | PySide6 (Qt6)               |
| Speech-to-text   | faster-whisper (medium)     |
| Voice detection  | Silero VAD                  |
| AI tutor         | Ollama (qwen2.5:7b)         |
| Text-to-speech   | VOICEVOX                    |
| Grammar check    | GiNZA (spaCy-based)         |
| Memory           | SQLite (local file)         |

---

## Prerequisites

### 1. System packages

```bash
# Ubuntu / Debian
sudo apt install ffmpeg python3-pip

# Arch Linux
sudo pacman -S ffmpeg python-pip

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

### 2. Python packages

```bash
pip install -r requirements.txt
```

> GiNZA installs a ~100MB Japanese NLP model. It's optional but recommended.  
> The app works without it — accuracy warnings just won't be as detailed.

### 3. Ollama

Install from https://ollama.com then pull the model:

```bash
# Start the server (keep this running)
ollama serve

# Pull the recommended model (~4.7 GB download)
ollama pull qwen2.5:7b

# Alternative if you want more speed over quality
ollama pull qwen2.5:3b
```

### 4. VOICEVOX

1. Download from https://voicevox.hiroshiba.jp  
2. Install and launch it — it runs as a background server on port 50021  
3. Keep it running while using the tutor

---

## Configuration

Edit `config.py` to change:

| Setting               | Default         | Notes                              |
|-----------------------|-----------------|------------------------------------|
| `OLLAMA_MODEL`        | `qwen2.5:7b`    | Any Ollama model with Japanese support |
| `WHISPER_MODEL`       | `medium`        | `small` is faster, `large-v3` is more accurate |
| `VOICEVOX_SPEAKER_ID` | `1` (ずんだもん) | See speaker list below             |
| `VAD_THRESHOLD`       | `0.5`           | Raise if mic picks up too much noise |
| `SILENCE_SECONDS`     | `1.2`           | How long to wait before processing speech |

### VOICEVOX speaker IDs (common)

| ID | Character    |
|----|--------------|
| 0  | 四国めたん    |
| 1  | ずんだもん    |
| 2  | 春日部つむぎ  |
| 3  | 雨晴はう     |
| 8  | 冥鳴ひまり   |
| 13 | 青山龍星     |

Run `GET http://localhost:50021/speakers` for the full list.

---

## Running

```bash
# Make sure Ollama is running: ollama serve
# Make sure VOICEVOX is running (launch the app)

python main.py
```

The app will:
1. Load Whisper (first launch takes ~30s)
2. Connect to Ollama and VOICEVOX (green dots in header)
3. Start listening immediately

---

## First launch checklist

- [ ] `ollama serve` is running in a terminal
- [ ] `qwen2.5:7b` (or your chosen model) is pulled
- [ ] VOICEVOX app is open and running
- [ ] Python dependencies are installed
- [ ] ffmpeg is installed
- [ ] Microphone is connected and working

---

## Troubleshooting

**Ollama dot is red**  
→ Run `ollama serve` in a terminal and keep it open

**VOICEVOX dot is red**  
→ Open the VOICEVOX application. It starts its server automatically.

**No transcription / always empty**  
→ Check your microphone is set as default input  
→ Try lowering `VAD_THRESHOLD` in config.py (e.g. 0.3)

**Tutor responds very slowly (30+ seconds)**  
→ Normal on CPU. Try `qwen2.5:3b` for faster but slightly lower quality

**Japanese fonts look wrong**  
→ Install Noto Sans JP: `sudo apt install fonts-noto-cjk`

---

## Memory & data

All data is stored locally in `japanese_tutor.db` (SQLite).  
- Conversation history  
- Mistake log  
- Your style profile (level, grammar notes, vocabulary notes)

The tutor reads this on every session — it genuinely remembers you.  
Delete `japanese_tutor.db` to start fresh.

---

## Accuracy warning system

The ⚠ flag appears when GiNZA detects a structural grammar issue,  
or when the tutor's response contains uncertainty markers.  
It does **not** mean the Japanese is definitely wrong — always use your own judgment  
or cross-check with a dictionary/grammar reference.

Recommended references:
- https://jisho.org (dictionary)
- https://bunpro.jp (grammar)
- https://www.nhk.or.jp/lesson (NHK Japanese lessons)