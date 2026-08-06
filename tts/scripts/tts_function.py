"""
tts_function.py — Reusable TTS module (MMS-TTS, Assamese)
==========================================================
Model   : facebook/mms-tts-asm  (Meta MMS · VITS · Assamese-specific)
No gating required — works out of the box via transformers VitsModel.

The model and tokeniser are loaded ONCE at module level (lazy singleton)
the first time text_to_speech() is called. Subsequent calls reuse the
already-loaded objects — no re-download, no re-initialisation.

Usage:
    from tts.scripts.tts_function import text_to_speech
    text_to_speech("নমস্কাৰ, আপুনি কেনে আছে?", "outputs/hello.wav")
"""

import os
import sys
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# UTF-8 stdout for Assamese script output on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # tts/scripts/
TTS_DIR      = os.path.dirname(SCRIPT_DIR)                  # tts/
PROJECT_ROOT = os.path.dirname(TTS_DIR)                     # project root
DEFAULT_OUTPUTS_DIR = os.path.join(TTS_DIR, "outputs")      # tts/outputs/

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_ID  = "facebook/mms-tts-asm"

# ---------------------------------------------------------------------------
# Lazy singleton — populated on first call to text_to_speech()
# ---------------------------------------------------------------------------
_model:     Optional[object] = None
_tokenizer: Optional[object] = None


def _load_model() -> Tuple[object, object]:
    """Load MMS-TTS Assamese model and tokeniser (cached globally after first call)."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    try:
        from transformers import VitsModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers is required. "
            "Install with: pip install transformers"
        )

    print(f"[TTS] Loading tokeniser '{MODEL_ID}' …", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[TTS] Loading model '{MODEL_ID}' …", flush=True)
    _model = VitsModel.from_pretrained(MODEL_ID)
    _model.eval()

    print("[TTS] Model ready.", flush=True)
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_to_speech(text: str, output_path: str) -> None:
    """
    Synthesise Assamese speech from text and save to a WAV file.

    Parameters
    ----------
    text : str
        Assamese text to synthesise (Unicode / Bengali script).
    output_path : str
        Destination path for the output WAV file.
        Parent directories are created automatically if they don't exist.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If text is empty.
    RuntimeError
        If model inference or file I/O fails.
    """
    import torch
    import numpy as np
    import soundfile as sf

    if not text or not text.strip():
        raise ValueError("text must be a non-empty string.")

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer = _load_model()

    # Tokenise
    inputs = tokenizer(text, return_tensors="pt")

    # Infer
    with torch.no_grad():
        output = model(**inputs)

    # VitsModel returns a ModelOutput; waveform shape is (1, num_samples)
    waveform = output.waveform.squeeze().numpy().astype(np.float32)
    sr       = model.config.sampling_rate

    # Normalise to prevent clipping
    peak = float(abs(waveform).max())
    if peak > 0:
        waveform /= peak

    sf.write(output_path, waveform, sr)

    duration = len(waveform) / sr
    print(f"[TTS] Saved → {output_path}")
    print(f"      Model       : {MODEL_ID}")
    print(f"      Sample rate : {sr} Hz")
    print(f"      Duration    : {duration:.2f} s")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SAMPLE_TEXT   = "নমস্কাৰ, আপুনি কেনে আছে?"
    SAMPLE_OUTPUT = os.path.join(DEFAULT_OUTPUTS_DIR, "tts_function_test.wav")

    print("=" * 60)
    print("  TTS self-test — MMS-TTS · Assamese")
    print("=" * 60)
    print(f"\nInput text  : {SAMPLE_TEXT}")
    print(f"Output file : {SAMPLE_OUTPUT}\n")

    text_to_speech(SAMPLE_TEXT, SAMPLE_OUTPUT)

    print("\n✅ Self-test complete.")
