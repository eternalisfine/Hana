# tests/conftest.py — Shared test fixtures, mock helpers, and test base classes

import unittest
from unittest.mock import patch, MagicMock
import sqlite3
import numpy as np
import io
from typing import Optional, List, Dict, Any


# ── SQLite In-Memory Database Base Class ──────────────────────────────────────

class InMemoryDBTestCase(unittest.TestCase):
    """
    Base TestCase that isolates all database operations into an in-memory SQLite DB.
    Guarantees zero writes to the real japanese_tutor.db file.
    """
    def setUp(self):
        import threading
        self._db_lock = threading.Lock()
        self.conn = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        
        test_self = self
        class MockCM:
            def __init__(self, conn):
                self.conn = conn
            def __enter__(self):
                test_self._db_lock.acquire()
                return self.conn
            def __exit__(self, exc_type, exc_val, exc_tb):
                test_self._db_lock.release()

        self.conn_patcher = patch("memory._conn", return_value=MockCM(self.conn))
        self.mock_conn_cm = self.conn_patcher.start()

        import memory
        memory.init_db()

    def tearDown(self):
        self.conn.close()
        self.conn_patcher.stop()


# ── Mock spaCy Doc & Token Helpers ───────────────────────────────────────────

class MockMorph:
    """Simulates spaCy's Token.morph object."""
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data or {}

    def get(self, key: str, default=None):
        val = self._data.get(key, default)
        if val is not None and not isinstance(val, (list, tuple)):
            return [val]
        return val


class MockToken:
    """Simulates a spaCy Token."""
    def __init__(self, text: str, pos: str = "NOUN", lemma: Optional[str] = None,
                 reading: Optional[str] = None, morph_data: Optional[Dict] = None):
        self.text = text
        self.pos_ = pos
        self.lemma_ = lemma if lemma is not None else text
        morph_dict = morph_data or {}
        if reading:
            morph_dict["Reading"] = [reading]
        self.morph = MockMorph(morph_dict)

    def __repr__(self):
        return f"MockToken({self.text!r}, pos={self.pos_!r}, lemma={self.lemma_!r})"


class MockSpan:
    """Simulates a spaCy Span (sentence)."""
    def __init__(self, tokens: List[MockToken], text: Optional[str] = None):
        self.tokens = tokens
        self.text = text if text is not None else "".join(t.text for t in tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


class MockDoc:
    """Simulates a spaCy Doc."""
    def __init__(self, tokens: List[MockToken], sents: Optional[List[MockSpan]] = None):
        self.tokens = tokens
        self.text = "".join(t.text for t in tokens)
        if sents is not None:
            self.sents = sents
        else:
            self.sents = [MockSpan(tokens, self.text)]

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def create_mock_doc(token_specs: List[Dict[str, Any]], sents_specs: Optional[List[List[int]]] = None) -> MockDoc:
    """
    Factory to create a MockDoc from token specifications.
    token_specs: list of dicts, e.g. [{"text": "私", "pos": "PRON", "lemma": "私", "reading": "ワタシ"}]
    sents_specs: optional list of token index ranges for multiple sentences, e.g. [[0, 3], [3, 6]]
    """
    tokens = [
        MockToken(
            text=spec["text"],
            pos=spec.get("pos", "NOUN"),
            lemma=spec.get("lemma", spec["text"]),
            reading=spec.get("reading", None),
            morph_data=spec.get("morph_data", None)
        )
        for spec in token_specs
    ]
    if sents_specs:
        sents = []
        for sent_indices in sents_specs:
            sent_tokens = [tokens[i] for i in sent_indices if i < len(tokens)]
            sents.append(MockSpan(sent_tokens))
        return MockDoc(tokens, sents=sents)
    return MockDoc(tokens)


# ── Synthetic Audio Helper ───────────────────────────────────────────────────

def sample_audio(duration: float = 1.0, sample_rate: int = 16000,
                 kind: str = "sine", freq: float = 440.0) -> np.ndarray:
    """
    Generate synthetic float32 mono audio array.
    kind: 'silence', 'sine', 'loud', 'noise'
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)

    if kind == "silence":
        return np.zeros(num_samples, dtype=np.float32)
    elif kind == "sine":
        return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    elif kind == "loud":
        return (0.9 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    elif kind == "noise":
        return (0.02 * np.random.randn(num_samples)).astype(np.float32)
    else:
        return np.zeros(num_samples, dtype=np.float32)


# ── Mock HTTP Response Helper ────────────────────────────────────────────────

class MockHTTPResponse:
    """Simulates a requests.Response object."""
    def __init__(self, status_code: int = 200, json_data: Any = None,
                 text: str = "", content: bytes = b""):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.text = text or (str(json_data) if json_data else "")
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}: {self.text}")
