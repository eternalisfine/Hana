# nlp_core.py — Shared NLP Initialization and Thread-Safe Access
import threading
from typing import Optional

_nlp_lock = threading.Lock()
_nlp_instance = None
_ginza_ok = False
_initialized = False

def _init_ginza():
    global _nlp_instance, _ginza_ok, _initialized
    if _initialized:
        return

    try:
        import spacy
        from spacy.util import registry
        
        # FIX FOR Python 3.14 + Pydantic v2 / Confection compatibility
        # Re-register compound_splitter with Optional[str] before loading ja_ginza
        if "compound_splitter" not in registry.factories:
            pass # It should be registered, but if not we can just define it
            
        from ginza.compound_splitter import CompoundSplitter

        def make_compound_splitter_fixed(
            nlp: spacy.language.Language,
            name: str,
            split_mode: Optional[str] = None,
        ):
            return CompoundSplitter(nlp.vocab, split_mode)

        # Overwrite the broken factory in the registry
        registry.factories.register("compound_splitter", func=make_compound_splitter_fixed)

        _nlp_instance = spacy.load("ja_ginza")
        _ginza_ok = True
    except Exception as e:
        print(f"[NLP] Failed to load GiNZA: {e}")
        _nlp_instance = None
        _ginza_ok = False
    finally:
        _initialized = True

def get_nlp():
    """Returns (nlp_instance, ginza_ok) safely initialized."""
    with _nlp_lock:
        if not _initialized:
            _init_ginza()
        return _nlp_instance, _ginza_ok

def process_text_safely(text: str):
    """Processes text in a thread-safe manner since spaCy docs are not thread-safe if shared, 
    but parsing creates independent docs. 
    Using the lock globally just for inference is safe against model mutation."""
    nlp, ok = get_nlp()
    if not ok or not nlp:
        return None
    
    with _nlp_lock:
        try:
            return nlp(text)
        except Exception as e:
            print(f"[NLP] Error processing text: {e}")
            return None
