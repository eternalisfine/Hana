# main.py — PySide6 desktop app (Enhanced UI Edition - Rock Solid Stability)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # this line hides gpu from pytorch

import sys
import uuid
import queue
import threading
import numpy as np
import time
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QScrollArea, QFrame, QPushButton, QDialog, QFormLayout,
    QLineEdit, QSpinBox, QComboBox, QSizePolicy, QSplitter,
    QTextEdit, QGraphicsDropShadowEffect, QSlider, QGraphicsOpacityEffect,
    QStackedWidget, QGraphicsBlurEffect, QGraphicsEffect,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer, QSize, QPropertyAnimation, QRect, QEasingCurve, QPoint, Property
from PySide6.QtGui import QFont, QColor, QPalette, QPixmap, QIcon, QFontDatabase, QLinearGradient, QBrush, QPainter, QPainterPath, QPen, QRadialGradient

import memory
import stt
import tutor
import safety
from listener import VoiceListener
from tts import TTSPlayer
from config import OLLAMA_MODEL, VOICEVOX_SPEAKER_ID

# Custom exit code for handling safe restarts
EXIT_CODE_REBOOT = -123


# ── Enhanced Colour palette (2026 Dark Glassmorphism) ─────────────────────────

BG           = "#0a0a12"         
BG_PANEL     = "#12121c"         
BG_CARD      = "rgba(24, 24, 36, 0.85)"  
BG_CARD_SOLID = "#181824"        
BG_USER      = "rgba(26, 42, 58, 0.9)"
BG_TUTOR     = "rgba(28, 28, 44, 0.9)"
BG_GLASS     = "rgba(30, 30, 48, 0.6)"
BG_GLASS_BORDER = "rgba(255, 255, 255, 0.08)"
ACCENT       = "#8b7cf7"         
ACCENT_GLOW  = "#a594ff"         
ACCENT_DARK  = "#6c5ce7"
ACCENT_GRAD_START = "#9d8eff"
ACCENT_GRAD_END   = "#7c6af5"
WARN         = "#ff8c42"
SUCCESS      = "#4ade80"
SUCCESS_GLOW = "#6ee7a0"
INFO         = "#38bdf8"
TEXT_PRI     = "#f1f1f7"
TEXT_SEC     = "#9ca3af"
TEXT_FADED   = "#4b5563"
BORDER       = "rgba(255, 255, 255, 0.06)"
BORDER_ACTIVE = "rgba(139, 124, 247, 0.3)"

STATUS_COLORS = {
    "loading":      "#a855f7", # Purple for model loading
    "listening":    SUCCESS,
    "recording":    "#f87171",
    "transcribing": "#fbbf24",
    "thinking":     ACCENT,
    "speaking":     INFO,
    "error":        "#f87171",
    "idle":         TEXT_FADED,
}

STATUS_LABELS = {
    "loading":      "◉ Loading AI Models...",
    "listening":    "● Listening",
    "recording":    "● Recording",
    "transcribing": "◉ Transcribing...",
    "thinking":     "◉ Thinking...",
    "speaking":     "◎ Speaking",
    "error":        "✕ Error",
    "idle":         "○ Idle",
}


# ── Glassmorphism helper widget ──────────────────────────────────────────────

class GlassPanel(QFrame):
    def __init__(self, parent=None, radius=16, glow_color=None):
        super().__init__(parent)
        self.radius = radius
        self.glow_color = glow_color or ACCENT
        self._hover = False
        self.setMouseTracking(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), self.radius, self.radius)

        painter.fillPath(path, QColor(BG_CARD_SOLID))

        pen = QPen(QColor(BORDER))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)

        highlight = QLinearGradient(0, 0, 0, self.height() * 0.4)
        highlight.setColorAt(0, QColor("rgba(255, 255, 255, 0.05)"))
        highlight.setColorAt(1, QColor("rgba(255, 255, 255, 0)"))
        painter.fillPath(path, highlight)

        painter.end()


# ── Animated status indicator ───────────────────────────────────────────

class PulsingDot(QWidget):
    def __init__(self, parent=None, size=10):
        super().__init__(parent)
        self.setFixedSize(size * 4, size * 4)
        self._size = size
        self._color = QColor(TEXT_FADED)
        self._pulse_anim = None
        self._pulse_scale = 1.0

    def get_pulse_scale(self):
        return self._pulse_scale

    def set_pulse_scale(self, val):
        self._pulse_scale = val
        self.update()

    pulse_scale = Property(float, get_pulse_scale, set_pulse_scale)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def start_pulse(self):
        if self._pulse_anim is None:
            self._pulse_anim = QPropertyAnimation(self, b"pulse_scale", self)
            self._pulse_anim.setDuration(1500)
            self._pulse_anim.setStartValue(1.0)
            self._pulse_anim.setEndValue(1.5)
            self._pulse_anim.setEasingCurve(QEasingCurve.InOutSine)
            self._pulse_anim.setLoopCount(-1)
        self._pulse_anim.start()

    def stop_pulse(self):
        if self._pulse_anim:
            self._pulse_anim.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()

        glow_radius = self._size * 2 * self._pulse_scale
        glow = QRadialGradient(center, glow_radius)
        glow.setColorAt(0, QColor(self._color.red(), self._color.green(), self._color.blue(), int(60 / self._pulse_scale)))
        glow.setColorAt(1, QColor(self._color.red(), self._color.green(), self._color.blue(), 0))
        
        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, glow_radius, glow_radius)

        painter.setBrush(self._color)
        painter.drawEllipse(center, self._size, self._size)
        painter.end()


# ── Worker signals ────────────────────────────────────────────────────────────

class Signals(QObject):
    status_changed    = Signal(str)           
    user_message      = Signal(str)           
    tutor_message     = Signal(str, bool)     
    error_message     = Signal(str)
    connection_status = Signal(str, bool)
    teardown_complete = Signal(int) # Emitted when safe to close app


# ── Processing pipeline (runs in background thread) ─────────────────────────

class Pipeline(QThread):
    def __init__(self, audio_queue: queue.Queue, signals: Signals, session_id: str):
        super().__init__()
        self.audio_queue = audio_queue
        self.signals     = signals
        self.session_id  = session_id
        self.tts         = TTSPlayer()
        self._running    = True

    def run(self):
        # 1. Load heavy models in the background to prevent UI freeze
        self.signals.status_changed.emit("loading")
        try:
            stt.load_model()
        except Exception as e:
            self.signals.error_message.emit(f"Failed to load STT: {e}")
        
        self.signals.status_changed.emit("listening")

        # 2. Start normal pipeline processing
        while self._running:
            try:
                audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not self._running:
                break

            self.tts.interrupt()

            self.signals.status_changed.emit("transcribing")
            text = stt.transcribe(audio)

            if not text.strip():
                self.signals.status_changed.emit("listening")
                continue

            self.signals.user_message.emit(text)
            self.signals.status_changed.emit("thinking")
            
            result = tutor.chat(text, self.session_id)

            if result["error"]:
                self.signals.error_message.emit(result["response"])
                self.signals.status_changed.emit("error")
                continue

            response = result["response"]
            safety_result = safety.check(response)
            flagged = safety_result["flagged"]

            self.signals.tutor_message.emit(response, flagged)

            self.signals.status_changed.emit("speaking")
            japanese_text = tutor.extract_japanese_for_tts(response)
            
            self.tts.on_end = lambda: self.signals.status_changed.emit("listening")
            self.tts.speak(japanese_text)


# ── Message bubble widget ─────────────────────────────────────────────────────

class MessageBubble(QFrame):
    def __init__(self, role: str, text: str, flagged: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("bubble")

        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = None
        self._en_fade_anim = None

        is_user = role == "user"
        outer = QHBoxLayout(self)
        outer.setContentsMargins(12, 4, 12, 4)

        bubble = GlassPanel(radius=18)
        bubble.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        if role == "assistant":
            content_widget = self._build_assistant_content(text, flagged)
        else:
            content_widget = self._build_user_content(text)

        inner_layout = QVBoxLayout(bubble)
        inner_layout.setContentsMargins(16, 12, 16, 12)
        inner_layout.setSpacing(6)
        inner_layout.addWidget(content_widget)

        bg = BG_USER if is_user else BG_TUTOR
        border_color = WARN if flagged else BORDER

        bubble.setStyleSheet(f"""
            GlassPanel {{
                background: {bg};
                border: 1px solid {border_color};
                border-radius: 18px;
            }}
        """)

        if is_user:
            outer.addStretch()
            outer.addWidget(bubble)
        else:
            outer.addWidget(bubble)
            outer.addStretch()

        bubble.setMaximumWidth(680)
        QTimer.singleShot(50, self._animate_entrance)

    def _animate_entrance(self):
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setDuration(400)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fade_anim.start()

    def set_english_visible(self, visible: bool):
        container = getattr(self, '_english_container', None)
        if container is not None:
            effect = QGraphicsOpacityEffect(container)
            container.setGraphicsEffect(effect)
            
            self._en_fade_anim = QPropertyAnimation(effect, b"opacity", self)
            self._en_fade_anim.setDuration(200)
            self._en_fade_anim.setStartValue(0.0 if visible else 1.0)
            self._en_fade_anim.setEndValue(1.0 if visible else 0.0)
            
            if visible:
                container.setVisible(True)
                self._en_fade_anim.start()
            else:
                self._en_fade_anim.finished.connect(lambda: container.setVisible(False))
                self._en_fade_anim.start()

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

        parts = text.split("---", 1)
        japanese_part = parts[0].strip()
        meta_part     = parts[1].strip() if len(parts) > 1 else ""

        jp_lbl = QLabel(japanese_part)
        jp_lbl.setWordWrap(True)
        jp_lbl.setFont(QFont("Noto Sans JP", 14))
        jp_lbl.setStyleSheet(f"color: {TEXT_PRI}; background: transparent;")
        jp_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(jp_lbl)

        self._english_container = QWidget()
        en_layout = QVBoxLayout(self._english_container)
        en_layout.setContentsMargins(0, 4, 0, 0)
        en_layout.setSpacing(4)

        if meta_part:
            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color: {BORDER}; background: {BORDER}; max-height:1px;")
            en_layout.addWidget(sep)
            meta_lbl = QLabel(meta_part)
            meta_lbl.setWordWrap(True)
            meta_lbl.setFont(QFont("Noto Sans JP", 11))
            meta_lbl.setStyleSheet(f"color: {TEXT_SEC}; background: transparent;")
            meta_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            en_layout.addWidget(meta_lbl)
        else:
            self._english_container.setVisible(False)

        layout.addWidget(self._english_container)

        if flagged:
            warn_frame = QFrame()
            warn_frame.setStyleSheet(f"""
                QFrame {{
                    background: rgba(255, 140, 66, 0.1);
                    border: 1px solid rgba(255, 140, 66, 0.3);
                    border-radius: 8px;
                    padding: 6px 10px;
                }}
            """)
            warn_layout = QHBoxLayout(warn_frame)
            warn_layout.setContentsMargins(10, 6, 10, 6)

            warn_icon = QLabel("⚠")
            warn_icon.setFont(QFont("Noto Sans JP", 12))
            warn_lbl = QLabel("Accuracy flagged — verify this phrasing")
            warn_lbl.setFont(QFont("Noto Sans JP", 10))
            warn_lbl.setStyleSheet(f"color: {WARN}; background: transparent;")

            warn_layout.addWidget(warn_icon)
            warn_layout.addWidget(warn_lbl)
            warn_layout.addStretch()
            layout.addWidget(warn_frame)

        return w


# ── Scrollable chat area ─────────────────────────────────────────

class ChatArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(f"""
            QScrollArea {{ background: {BG}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 8px; border-radius: 4px; }}
            QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 40px; }}
            QScrollBar::handle:vertical:hover {{ background: {ACCENT}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        """)

        self._container   = QWidget()
        self._layout      = QVBoxLayout(self._container)
        self._bubbles     = []
        self._show_english = True
        self._layout.setContentsMargins(0, 16, 0, 16)
        self._layout.setSpacing(8)
        self._layout.addStretch()
        self.setWidget(self._container)

    def add_message(self, role: str, text: str, flagged: bool = False):
        bubble = MessageBubble(role, text, flagged)
        bubble.set_english_visible(self._show_english)
        self._bubbles.append(bubble)
        self._layout.insertWidget(self._layout.count() - 1, bubble)
        QTimer.singleShot(50, self._scroll_to_bottom)

    def set_english_visible(self, visible: bool):
        self._show_english = visible
        for b in self._bubbles:
            b.set_english_visible(visible)

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
        layout.setSpacing(8)

        self._dot = PulsingDot(size=5)
        self._lbl = QLabel(label)
        self._lbl.setFont(QFont("Noto Sans JP", 10))
        self._lbl.setStyleSheet(f"color: {TEXT_SEC};")

        layout.addWidget(self._dot)
        layout.addWidget(self._lbl)
        self.set_ok(False)

    def set_ok(self, ok: bool):
        color = SUCCESS if ok else TEXT_FADED
        self._dot.set_color(color)
        if ok:
            self._dot.start_pulse()
        else:
            self._dot.stop_pulse()


# ── Settings dialog ───────────────────────────────────────────────────────────

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(420)
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_PANEL}; color: {TEXT_PRI}; border: 1px solid {BORDER}; border-radius: 16px; }}
            QLabel  {{ color: {TEXT_PRI}; font-weight: 500; }}
            QLineEdit, QSpinBox, QComboBox {{ background: {BG_CARD_SOLID}; color: {TEXT_PRI}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 12px; font-size: 13px; }}
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
            QPushButton {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_GRAD_START}, stop:1 {ACCENT_GRAD_END}); color: white; border: none; border-radius: 8px; padding: 10px 20px; font-weight: 600; }}
            QPushButton:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_GLOW}, stop:1 {ACCENT}); }}
        """)

        from config import OLLAMA_MODEL, VOICEVOX_SPEAKER_ID, WHISPER_MODEL
        layout = QFormLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(24, 24, 24, 24)

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
        import config, os, sys
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        with open(config_path, "r") as f:
            src = f.read()

        def _replace(src, key, value):
            import re
            if isinstance(value, str):
                return re.sub(rf'^({key}\s*=\s*).*', rf'\g<1>"{value}"', src, flags=re.MULTILINE)
            else:
                return re.sub(rf'^({key}\s*=\s*).*', rf'\g<1>{value}', src, flags=re.MULTILINE)

        src = _replace(src, "OLLAMA_MODEL",        self._model.text().strip())
        src = _replace(src, "WHISPER_MODEL",        self._whisper.text().strip())
        src = _replace(src, "VOICEVOX_SPEAKER_ID",  self._speaker.value())

        with open(config_path, "w") as f:
            f.write(src)

        # Flag the parent window that we want to reboot, then close it cleanly
        parent = self.parent()
        if parent:
            parent._reboot_requested = True
            parent.close()
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
        self._init_connection_checks() 

    def _init_db(self):
        memory.init_db()
        # STT model load was moved to the Pipeline thread to prevent GUI startup freezes!

    def _build_ui(self):
        self.setWindowTitle("Japanese Tutor — はな")
        self.resize(900, 720)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        central = QWidget()
        central.setStyleSheet(f"background: {BG};")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = GlassPanel(radius=0)
        header.setFixedHeight(64)
        header.setStyleSheet(f"GlassPanel {{ background: {BG_PANEL}; border-bottom: 1px solid {BORDER}; }}")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(24, 0, 20, 0)

        icon_lbl = QLabel("🌸")
        icon_lbl.setFont(QFont("Noto Sans JP", 20))
        title = QLabel("はな")
        title.setFont(QFont("Noto Sans JP", 17, QFont.Bold))
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
        settings_btn.setFixedSize(36, 36)
        settings_btn.setFont(QFont("monospace", 14))
        settings_btn.setStyleSheet(f"""
            QPushButton {{ background: {BG_CARD_SOLID}; color: {TEXT_SEC}; border: 1px solid {BORDER}; border-radius: 10px; }}
            QPushButton:hover {{ background: {ACCENT}; color: white; border: none; }}
        """)
        settings_btn.clicked.connect(self._open_settings)

        h_layout.addWidget(icon_lbl)
        h_layout.addSpacing(8)
        h_layout.addLayout(title_block)
        h_layout.addStretch()
        h_layout.addWidget(self._ollama_dot)
        h_layout.addSpacing(20)
        h_layout.addWidget(self._vv_dot)
        h_layout.addSpacing(16)
        h_layout.addWidget(settings_btn)

        self._chat = ChatArea()

        status_bar = GlassPanel(radius=0)
        status_bar.setFixedHeight(48)
        status_bar.setStyleSheet(f"GlassPanel {{ background: {BG_PANEL}; border-top: 1px solid {BORDER}; }}")
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(24, 0, 24, 0)

        self._status_lbl = QLabel(STATUS_LABELS["loading"]) # Set initial status
        self._status_lbl.setFont(QFont("Noto Mono", 11))
        self._status_lbl.setStyleSheet(f"color: {STATUS_COLORS['loading']}; background: transparent;")

        self._session_lbl = QLabel(f"Session  {self.session_id[:8]}")
        self._session_lbl.setFont(QFont("Noto Mono", 9))
        self._session_lbl.setStyleSheet(f"color: {TEXT_FADED}; background: transparent;")

        sb_layout.addWidget(self._status_lbl)
        sb_layout.addStretch()
        sb_layout.addWidget(self._session_lbl)

        root.addWidget(header)
        root.addWidget(self._build_controls_bar())
        root.addWidget(self._chat)
        root.addWidget(status_bar)

    def _build_controls_bar(self) -> QFrame:
        bar = QFrame()
        bar.setFixedHeight(48)
        bar.setStyleSheet(f"QFrame {{ background: {BG_PANEL}; border-bottom: 1px solid {BORDER}; }}")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        self._en_btn = QPushButton("🇬🇧  English  ON")
        self._en_btn.setCheckable(True)
        self._en_btn.setChecked(True)
        self._en_btn.setFixedHeight(30)
        self._en_btn.setFont(QFont("Noto Sans JP", 10))
        self._en_btn.setStyleSheet(
            f"QPushButton {{ background: {ACCENT}; color: white; border: none; border-radius: 8px; padding: 0 14px; font-weight: 500; }}"
            f"QPushButton:!checked {{ background: {BG_CARD_SOLID}; color: {TEXT_SEC}; border: 1px solid {BORDER}; }}"
            f"QPushButton:hover:!checked {{ background: {BG_USER}; color: {TEXT_PRI}; }}"
        )
        self._en_btn.toggled.connect(self._on_en_toggle)

        self._pause_btn = QPushButton("⏸  Pause")
        self._pause_btn.setCheckable(True)
        self._pause_btn.setChecked(False)
        self._pause_btn.setFixedHeight(30)
        self._pause_btn.setFont(QFont("Noto Sans JP", 10))
        self._pause_btn.setStyleSheet(
            f"QPushButton {{ background: {BG_CARD_SOLID}; color: {TEXT_SEC}; border: 1px solid {BORDER}; border-radius: 8px; padding: 0 14px; font-weight: 500; }}"
            f"QPushButton:checked {{ background: {WARN}; color: white; border: none; }}"
            f"QPushButton:hover:!checked {{ background: {BG_USER}; color: {TEXT_PRI}; }}"
        )
        self._pause_btn.toggled.connect(self._on_pause_toggle)

        speed_lbl = QLabel("Speed:")
        speed_lbl.setFont(QFont("Noto Sans JP", 10))
        speed_lbl.setStyleSheet(f"color: {TEXT_SEC}; background: transparent;")

        self._speed_val_lbl = QLabel("0.9x")
        self._speed_val_lbl.setFixedWidth(40)
        self._speed_val_lbl.setFont(QFont("Noto Mono", 10))
        self._speed_val_lbl.setStyleSheet(f"color: {TEXT_PRI}; background: transparent;")

        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setMinimum(5)
        self._speed_slider.setMaximum(20)
        self._speed_slider.setValue(9)
        self._speed_slider.setFixedWidth(160)
        self._speed_slider.setStyleSheet(
            f"QSlider::groove:horizontal {{ background: {BORDER}; height: 4px; border-radius: 2px; }}"
            f"QSlider::handle:horizontal {{ background: {ACCENT}; width: 16px; height: 16px; margin: -6px 0; border-radius: 8px; }}"
            f"QSlider::handle:horizontal:hover {{ background: {ACCENT_GLOW}; }}"
            f"QSlider::sub-page:horizontal {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {ACCENT_GRAD_START}, stop:1 {ACCENT_GRAD_END}); border-radius: 2px; }}"
        )
        self._speed_slider.valueChanged.connect(self._on_speed_change)

        layout.addWidget(self._en_btn)
        layout.addWidget(self._pause_btn)
        layout.addStretch()
        layout.addWidget(speed_lbl)
        layout.addWidget(self._speed_slider)
        layout.addWidget(self._speed_val_lbl)
        return bar

    def _connect_signals(self):
        s = self.signals
        s.status_changed.connect(self._on_status)
        s.user_message.connect(self._on_user_msg)
        s.tutor_message.connect(self._on_tutor_msg)
        s.error_message.connect(self._on_error)
        s.connection_status.connect(self._on_connection)
        
        # Connect the custom safe teardown signal
        s.teardown_complete.connect(lambda code: QApplication.instance().exit(code))

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
        if self._pipeline and self._pipeline.tts.is_speaking():
            self._pipeline.tts.interrupt()
        self.audio_queue.put(audio)

    def _init_connection_checks(self):
        self._conn_timer = QTimer(self)
        self._conn_timer.timeout.connect(self._run_connection_check)
        self._conn_timer.start(15_000)
        self._run_connection_check()

    def _run_connection_check(self):
        def _check():
            ok_ollama = tutor.check_ollama()
            ok_vv     = TTSPlayer().is_running()
            self.signals.connection_status.emit("ollama",   ok_ollama)
            self.signals.connection_status.emit("voicevox", ok_vv)
        threading.Thread(target=_check, daemon=True).start()

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
            if not ok: self._chat.add_system("Ollama not reachable — run: ollama serve")
        elif service == "voicevox":
            self._vv_dot.set_ok(ok)
            if not ok: self._chat.add_system("VOICEVOX not reachable — start the VOICEVOX app")

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()

    def _on_pause_toggle(self, paused: bool):
        if paused:
            self._pause_btn.setText("▶  Resume")
            if self._pipeline: self._pipeline.tts.interrupt()
            if self._listener: self._listener.mute()
            while not self.audio_queue.empty():
                try: self.audio_queue.get_nowait()
                except Exception: break
            self._on_status("idle")
            self._chat.add_system("⏸ Paused")
        else:
            self._pause_btn.setText("⏸  Pause")
            if self._listener: self._listener.unmute()
            self._on_status("listening")
            self._chat.add_system("▶ Resumed")

    def _on_en_toggle(self, checked: bool):
        self._en_btn.setText("🇬🇧  English  ON" if checked else "🇬🇧  English  OFF")
        self._chat.set_english_visible(checked)

    def _on_speed_change(self, value: int):
        speed = value / 10.0
        self._speed_val_lbl.setText(f"{speed:.1f}x")
        if self._pipeline:
            self._pipeline.tts.speed = speed

    def closeEvent(self, event):
        """Asynchronous shutdown to prevent 'Not Responding' UI freezes."""
        if getattr(self, "_shutting_down", False):
            event.accept()
            return

        self._shutting_down = True
        event.ignore()
        
        # 1. Hide the window instantly so the UX feels responsive
        self.hide()
        
        # 2. Safely stop the QTimer from the main thread
        if hasattr(self, '_conn_timer'):
            self._conn_timer.stop()

        # 3. Offload all the heavy cleanup to a background thread
        threading.Thread(target=self._perform_teardown, daemon=True).start()

    def _perform_teardown(self):
        """Runs in background: waits for AI models to safely release their memory."""
        if self._pipeline:
            self._pipeline._running = False
            self._pipeline.tts.interrupt()
            self._pipeline.wait(3000) # Give models 3s to gracefully exit
            
        if self._listener:
            self._listener.stop()
            
        # Determine exit code and signal the main thread to quit safely
        exit_code = EXIT_CODE_REBOOT if getattr(self, "_reboot_requested", False) else 0
        self.signals.teardown_complete.emit(exit_code)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Japanese Tutor")

    QFontDatabase.addApplicationFont("/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf")

    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(BG))
    palette.setColor(QPalette.WindowText,      QColor(TEXT_PRI))
    palette.setColor(QPalette.Base,            QColor(BG_CARD_SOLID))
    palette.setColor(QPalette.Text,            QColor(TEXT_PRI))
    palette.setColor(QPalette.Button,          QColor(BG_PANEL))
    palette.setColor(QPalette.ButtonText,      QColor(TEXT_PRI))
    palette.setColor(QPalette.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    win = MainWindow()
    win.show()

    # Capture the specific exit code from our teardown process
    exit_code = app.exec()

    # If the user clicked "Save & Restart", swap the process here!
    if exit_code == EXIT_CODE_REBOOT:
        os.execv(sys.executable, [sys.executable] + sys.argv)
    else:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()