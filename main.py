# main.py — PySide6 desktop app

import sys
import uuid
import queue
import threading
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QFrame, QPushButton, QDialog, QFormLayout,
    QLineEdit, QSpinBox, QComboBox, QSizePolicy, QSplitter,
    QTextEdit, QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QSize, QPropertyAnimation, QRect
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon, QFontDatabase

import memory
import stt
import tutor
import safety
from listener import VoiceListener
from tts import TTSPlayer
from config import OLLAMA_MODEL, VOICEVOX_SPEAKER_ID


# ── Colour palette ────────────────────────────────────────────────────────────

BG           = "#0f0f14"
BG_PANEL     = "#16161f"
BG_CARD      = "#1e1e2a"
BG_USER      = "#1a2a3a"
BG_TUTOR     = "#1e1e2a"
ACCENT       = "#7c6af5"         # Soft purple
ACCENT_DARK  = "#5a4fd4"
WARN         = "#e08030"
SUCCESS      = "#4caf80"
TEXT_PRI     = "#e8e8f0"
TEXT_SEC     = "#8888aa"
TEXT_FADED   = "#55556a"
BORDER       = "#2a2a3a"

STATUS_COLORS = {
    "listening":    "#4caf80",
    "recording":    "#e05050",
    "transcribing": "#e0a030",
    "thinking":     ACCENT,
    "speaking":     "#30a0e0",
    "error":        "#e05050",
    "idle":         TEXT_FADED,
}

STATUS_LABELS = {
    "listening":    "● Listening",
    "recording":    "● Recording",
    "transcribing": "◉ Transcribing...",
    "thinking":     "◉ Thinking...",
    "speaking":     "◎ Speaking",
    "error":        "✕ Error",
    "idle":         "○ Idle",
}


# ── Worker signals (cross-thread Qt communication) ────────────────────────────

class Signals(QObject):
    status_changed  = Signal(str)           # new status key
    user_message    = Signal(str)           # transcribed user text
    tutor_message   = Signal(str, bool)     # response text, flagged
    error_message   = Signal(str)
    connection_status = Signal(str, bool)   # service name, ok


# ── Processing pipeline (runs in background thread) ───────────────────────────

class Pipeline(QThread):
    def __init__(self, audio_queue: queue.Queue, signals: Signals, session_id: str):
        super().__init__()
        self.audio_queue = audio_queue
        self.signals     = signals
        self.session_id  = session_id
        self.tts         = TTSPlayer()
        self._running    = True

    def run(self):
        while self._running:
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # ── Interrupt TTS if speaking ──
            self.tts.interrupt()

            # ── Transcribe ────────────────
            self.signals.status_changed.emit("transcribing")
            text = stt.transcribe(audio)

            if not text.strip():
                self.signals.status_changed.emit("listening")
                continue

            self.signals.user_message.emit(text)

            # ── Tutor response ────────────
            self.signals.status_changed.emit("thinking")
            result = tutor.chat(text, self.session_id)

            if result["error"]:
                self.signals.error_message.emit(result["response"])
                self.signals.status_changed.emit("error")
                continue

            response = result["response"]

            # ── Safety check ──────────────
            safety_result = safety.check(response)
            flagged = safety_result["flagged"]

            self.signals.tutor_message.emit(response, flagged)

            # ── TTS ───────────────────────
            self.signals.status_changed.emit("speaking")
            japanese_text = tutor.extract_japanese_for_tts(response)
            self.tts.on_end = lambda: self.signals.status_changed.emit("listening")
            self.tts.speak(japanese_text)

    def stop(self):
        self._running = False
        self.tts.interrupt()


# ── Message bubble widget ─────────────────────────────────────────────────────

class MessageBubble(QFrame):
    def __init__(self, role: str, text: str, flagged: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("bubble")

        is_user = role == "user"

        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 4, 12, 4)

        bubble = QFrame()
        bubble.setObjectName("inner")
        bubble.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # Parse structured response into sections
        if role == "assistant":
            content_widget = self._build_assistant_content(text, flagged)
        else:
            content_widget = self._build_user_content(text)

        inner_layout = QVBoxLayout(bubble)
        inner_layout.setContentsMargins(14, 10, 14, 10)
        inner_layout.setSpacing(6)
        inner_layout.addWidget(content_widget)

        bg   = BG_USER if is_user else BG_TUTOR
        border_color = WARN if flagged else BORDER
        radius = 16

        bubble.setStyleSheet(f"""
            QFrame#inner {{
                background: {bg};
                border: 1px solid {border_color};
                border-radius: {radius}px;
            }}
        """)

        if is_user:
            outer.addStretch()
            outer.addWidget(bubble)
        else:
            outer.addWidget(bubble)
            outer.addStretch()

        # Max width
        bubble.setMaximumWidth(680)

    def _build_user_content(self, text: str) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Noto Sans JP", 13))
        lbl.setStyleSheet(f"color: {TEXT_PRI}; background: transparent;")
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(lbl)
        return w

    def _build_assistant_content(self, text: str, flagged: bool) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Split on --- separator
        parts = text.split("---", 1)
        japanese_part = parts[0].strip()
        meta_part     = parts[1].strip() if len(parts) > 1 else ""

        # Japanese text (larger)
        jp_lbl = QLabel(japanese_part)
        jp_lbl.setWordWrap(True)
        jp_lbl.setFont(QFont("Noto Sans JP", 14))
        jp_lbl.setStyleSheet(f"color: {TEXT_PRI}; background: transparent;")
        jp_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(jp_lbl)

        # Meta section (English, notes, corrections)
        if meta_part:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-height:1px;")
            layout.addWidget(sep)

            meta_lbl = QLabel(meta_part)
            meta_lbl.setWordWrap(True)
            meta_lbl.setFont(QFont("Noto Sans JP", 11))
            meta_lbl.setStyleSheet(f"color: {TEXT_SEC}; background: transparent;")
            meta_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(meta_lbl)

        # Warning badge
        if flagged:
            warn_lbl = QLabel("⚠ Accuracy flagged — verify this phrasing")
            warn_lbl.setFont(QFont("Noto Sans JP", 10))
            warn_lbl.setStyleSheet(
                f"color: {WARN}; background: transparent; padding-top: 4px;"
            )
            layout.addWidget(warn_lbl)

        return w


# ── Scrollable chat area ──────────────────────────────────────────────────────

class ChatArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:vertical {{ background: {BG}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
        """)

        self._container = QWidget()
        self._layout    = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 16, 0, 16)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        self.setWidget(self._container)

    def add_message(self, role: str, text: str, flagged: bool = False):
        bubble = MessageBubble(role, text, flagged)
        # Insert before the trailing stretch
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def add_system(self, text: str):
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Noto Sans JP", 10))
        lbl.setStyleSheet(f"color: {TEXT_FADED}; padding: 4px;")
        self._layout.insertWidget(self._layout.count() - 1, lbl)

    def _scroll_to_bottom(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


# ── Connection dot ────────────────────────────────────────────────────────────

class ConnectionDot(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self._dot = QLabel("●")
        self._dot.setFont(QFont("monospace", 10))
        self._lbl = QLabel(label)
        self._lbl.setFont(QFont("Noto Sans JP", 10))
        self._lbl.setStyleSheet(f"color: {TEXT_SEC};")

        layout.addWidget(self._dot)
        layout.addWidget(self._lbl)
        self.set_ok(False)

    def set_ok(self, ok: bool):
        color = SUCCESS if ok else TEXT_FADED
        self._dot.setStyleSheet(f"color: {color};")


# ── Settings dialog ───────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_PANEL}; color: {TEXT_PRI}; }}
            QLabel  {{ color: {TEXT_PRI}; }}
            QLineEdit, QSpinBox, QComboBox {{
                background: {BG_CARD}; color: {TEXT_PRI};
                border: 1px solid {BORDER}; border-radius: 6px; padding: 6px;
            }}
            QPushButton {{
                background: {ACCENT}; color: white;
                border: none; border-radius: 6px; padding: 8px 16px;
            }}
            QPushButton:hover {{ background: {ACCENT_DARK}; }}
        """)

        from config import OLLAMA_MODEL, VOICEVOX_SPEAKER_ID, WHISPER_MODEL
        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self._model  = QLineEdit(OLLAMA_MODEL)
        self._speaker = QSpinBox()
        self._speaker.setRange(0, 100)
        self._speaker.setValue(VOICEVOX_SPEAKER_ID)
        self._whisper = QLineEdit(WHISPER_MODEL)

        layout.addRow("Ollama model:", self._model)
        layout.addRow("VOICEVOX speaker ID:", self._speaker)
        layout.addRow("Whisper model:", self._whisper)

        save_btn = QPushButton("Save & restart")
        save_btn.clicked.connect(self._save)
        layout.addRow(save_btn)

    def _save(self):
        import config
        config.OLLAMA_MODEL = self._model.text().strip()
        config.VOICEVOX_SPEAKER_ID = self._speaker.value()
        config.WHISPER_MODEL = self._whisper.text().strip()
        self.accept()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.session_id  = str(uuid.uuid4())
        self.signals     = Signals()
        self.audio_queue = queue.Queue()
        self._pipeline   = None
        self._listener   = None

        self._init_db()
        self._build_ui()
        self._connect_signals()
        self._start_pipeline()
        self._start_listener()
        self._check_connections()

    # ── DB ────────────────────────────────────────────────────────────────────

    def _init_db(self):
        memory.init_db()
        stt.load_model()   # Pre-load Whisper

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("Japanese Tutor — はな")
        self.resize(860, 680)
        self.setMinimumSize(600, 400)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        central.setStyleSheet(f"background: {BG};")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header ──────────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background: {BG_PANEL};
                border-bottom: 1px solid {BORDER};
            }}
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 0, 16, 0)

        title = QLabel("はな / Hana")
        title.setFont(QFont("Noto Sans JP", 15, QFont.Bold))
        title.setStyleSheet(f"color: {TEXT_PRI}; background: transparent;")

        subtitle = QLabel("Japanese Conversation Tutor")
        subtitle.setFont(QFont("Noto Sans JP", 10))
        subtitle.setStyleSheet(f"color: {TEXT_SEC}; background: transparent;")

        title_block = QVBoxLayout()
        title_block.setSpacing(1)
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self._ollama_dot = ConnectionDot("Ollama")
        self._vv_dot     = ConnectionDot("VOICEVOX")

        settings_btn = QPushButton("⚙")
        settings_btn.setFixedSize(32, 32)
        settings_btn.setFont(QFont("monospace", 14))
        settings_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {TEXT_SEC};
                border: none; border-radius: 6px;
            }}
            QPushButton:hover {{ background: {BG_CARD}; color: {TEXT_PRI}; }}
        """)
        settings_btn.clicked.connect(self._open_settings)

        h_layout.addLayout(title_block)
        h_layout.addStretch()
        h_layout.addWidget(self._ollama_dot)
        h_layout.addSpacing(16)
        h_layout.addWidget(self._vv_dot)
        h_layout.addSpacing(12)
        h_layout.addWidget(settings_btn)

        # ── Chat area ────────────────────────────────────────────────────────
        self._chat = ChatArea()

        # ── Status bar ───────────────────────────────────────────────────────
        status_bar = QFrame()
        status_bar.setFixedHeight(44)
        status_bar.setStyleSheet(f"""
            QFrame {{
                background: {BG_PANEL};
                border-top: 1px solid {BORDER};
            }}
        """)
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(20, 0, 20, 0)

        self._status_lbl = QLabel(STATUS_LABELS["idle"])
        self._status_lbl.setFont(QFont("Noto Mono", 11))
        self._status_lbl.setStyleSheet(f"color: {STATUS_COLORS['idle']}; background: transparent;")

        self._session_lbl = QLabel(f"Session  {self.session_id[:8]}")
        self._session_lbl.setFont(QFont("Noto Mono", 9))
        self._session_lbl.setStyleSheet(f"color: {TEXT_FADED}; background: transparent;")

        sb_layout.addWidget(self._status_lbl)
        sb_layout.addStretch()
        sb_layout.addWidget(self._session_lbl)

        root.addWidget(header)
        root.addWidget(self._chat)
        root.addWidget(status_bar)

    # ── Signals ───────────────────────────────────────────────────────────────

    def _connect_signals(self):
        s = self.signals
        s.status_changed.connect(self._on_status)
        s.user_message.connect(self._on_user_msg)
        s.tutor_message.connect(self._on_tutor_msg)
        s.error_message.connect(self._on_error)
        s.connection_status.connect(self._on_connection)

    # ── Pipeline & listener ───────────────────────────────────────────────────

    def _start_pipeline(self):
        self._pipeline = Pipeline(self.audio_queue, self.signals, self.session_id)
        self._pipeline.start()

    def _start_listener(self):
        self._listener = VoiceListener(
            on_speech=self._on_audio,
            on_state_change=lambda s: self.signals.status_changed.emit(s),
        )
        threading.Thread(target=self._load_and_start_listener, daemon=True).start()

    def _load_and_start_listener(self):
        self._listener.load()
        self._listener.start()

    def _on_audio(self, audio: np.ndarray):
        """Called from listener thread — hand audio off to pipeline."""
        # Interrupt TTS via pipeline
        if self._pipeline and self._pipeline.tts.is_speaking():
            self._pipeline.tts.interrupt()
        self.audio_queue.put(audio)

    # ── Connection checks ─────────────────────────────────────────────────────

    def _check_connections(self):
        def _check():
            ok_ollama = tutor.check_ollama()
            ok_vv     = TTSPlayer().is_running()
            self.signals.connection_status.emit("ollama",   ok_ollama)
            self.signals.connection_status.emit("voicevox", ok_vv)

        threading.Thread(target=_check, daemon=True).start()
        # Re-check every 15s
        self._conn_timer = QTimer()
        self._conn_timer.timeout.connect(self._check_connections)
        self._conn_timer.start(15_000)

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_status(self, key: str):
        label = STATUS_LABELS.get(key, key)
        color = STATUS_COLORS.get(key, TEXT_SEC)
        self._status_lbl.setText(label)
        self._status_lbl.setStyleSheet(f"color: {color}; background: transparent;")

    def _on_user_msg(self, text: str):
        self._chat.add_message("user", text)

    def _on_tutor_msg(self, text: str, flagged: bool):
        self._chat.add_message("assistant", text, flagged)

    def _on_error(self, msg: str):
        self._chat.add_system(f"⚠ {msg}")

    def _on_connection(self, service: str, ok: bool):
        if service == "ollama":
            self._ollama_dot.set_ok(ok)
            if not ok:
                self._chat.add_system(
                    "Ollama not reachable — run: ollama serve"
                )
        elif service == "voicevox":
            self._vv_dot.set_ok(ok)
            if not ok:
                self._chat.add_system(
                    "VOICEVOX not reachable — start the VOICEVOX app"
                )

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._pipeline:
            self._pipeline.stop()
        if self._listener:
            self._listener.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Japanese Tutor")

    # Load Noto Sans JP if available
    QFontDatabase.addApplicationFont("/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf")

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT_PRI))
    palette.setColor(QPalette.Base,            QColor(BG_CARD))
    palette.setColor(QPalette.Text,            QColor(TEXT_PRI))
    palette.setColor(QPalette.Button,          QColor(BG_PANEL))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT_PRI))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()