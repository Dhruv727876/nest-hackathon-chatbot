"""
asr_function.py — Reusable ASR module (IndicConformer, Assamese)
================================================================
Model   : ai4bharat/indic-conformer-600m-multilingual
Language: Assamese ("as")

The model and tokeniser are loaded ONCE at module level (lazy singleton)
the first time speech_to_text() is called. Subsequent calls reuse the
already-loaded model — no re-download, no re-initialisation.

Usage:
    from asr.scripts.asr_function import speech_to_text
    text = speech_to_text("/path/to/audio.wav")
"""

import io
import os
import sys
from typing import Optional

import numpy as np

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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # asr/scripts/
ASR_DIR      = os.path.dirname(SCRIPT_DIR)                  # asr/
PROJECT_ROOT = os.path.dirname(ASR_DIR)                     # project root

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
MODEL_ID   = "ai4bharat/indic-conformer-600m-multilingual"
LANG_CODE  = "as"          # ISO 639-1 code for Assamese
DECODE_MODE = "ctc"
TARGET_SR  = 16_000        # IndicConformer expects 16 kHz mono

# ---------------------------------------------------------------------------
# Lazy singleton — populated on first call to speech_to_text()
# ---------------------------------------------------------------------------
_model: Optional[object] = None


def _get_hf_token() -> Optional[str]:
    """Read HF_TOKEN from environment or .env files."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    candidates = [
        os.path.join(PROJECT_ROOT, ".env"),
        os.path.join(PROJECT_ROOT, "nlu-mt", ".env"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("HF_TOKEN="):
                        val = line.split("=", 1)[1].strip().strip("\"'")
                        return val or None
        except Exception:
            pass
    return None


def _load_model() -> object:
    """Load IndicConformer from HuggingFace (or local cache). Cached globally."""
    global _model
    if _model is not None:
        return _model

    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers is required. "
            "Install with: pip install transformers onnxruntime"
        )

    token = _get_hf_token()
    print(f"[ASR] Loading model '{MODEL_ID}' …", flush=True)
    _model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=token,
    )
    print("[ASR] Model ready.", flush=True)
    return _model


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Linear-interpolation resampling (no external library needed)."""
    if orig_sr == target_sr:
        return audio
    duration       = len(audio) / orig_sr
    n_target       = int(round(duration * target_sr))
    x_orig         = np.linspace(0.0, duration, len(audio))
    x_target       = np.linspace(0.0, duration, n_target)
    return np.interp(x_target, x_orig, audio).astype(np.float32)


def _load_audio(audio_path: str) -> np.ndarray:
    """
    Load an audio file and return a float32 mono numpy array at TARGET_SR.
    Accepts any format supported by soundfile (WAV, FLAC, OGG …).
    Also accepts raw bytes stored in a dict (FLEURS decode=False format).
    """
    import soundfile as sf

    if isinstance(audio_path, dict):
        # Support FLEURS dataset dict: {"bytes": b"...", "path": "..."}
        raw = audio_path.get("bytes")
        if raw is None:
            raise ValueError("Audio dict has no 'bytes' key.")
        audio, sr = sf.read(io.BytesIO(raw))
    else:
        audio, sr = sf.read(audio_path)

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32)
    return _resample(audio, sr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speech_to_text(audio_path: str) -> str:
    """
    Transcribe audio to Assamese text using IndicConformer.

    Parameters
    ----------
    audio_path : str | dict
        Path to a WAV/FLAC/OGG file, or a FLEURS-style audio dict
        with a 'bytes' key.

    Returns
    -------
    str
        Transcribed text in Assamese script.

    Raises
    ------
    FileNotFoundError
        If audio_path is a string pointing to a non-existent file.
    RuntimeError
        If model inference fails.
    """
    import torch

    # Validate file path (skip for dict inputs)
    if isinstance(audio_path, str) and not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model()

    # Load + resample
    audio = _load_audio(audio_path)

    # Shape: (1, num_samples) — IndicConformer expects 2-D tensor
    wav_tensor = torch.tensor(audio).unsqueeze(0)

    with torch.no_grad():
        result = model(wav_tensor, LANG_CODE, DECODE_MODE)

    # Normalise output to str
    if isinstance(result, (list, tuple)):
        result = result[0] if result else ""
    return str(result).strip()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import datasets as hf_datasets

    print("=" * 60)
    print("  ASR self-test — IndicConformer · Assamese (FLEURS)")
    print("=" * 60)

    # Load cached FLEURS Assamese dataset (no re-download if already cached)
    print("\nLoading first sample from google/fleurs 'as_in' train split …")
    ds = hf_datasets.load_dataset("google/fleurs", "as_in", split="train")
    ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))
    sample = ds[0]

    ground_truth = sample.get("transcription", "")
    audio_dict   = sample["audio"]   # dict with 'bytes' key

    print(f"Ground truth : {ground_truth}")
    prediction = speech_to_text(audio_dict)
    print(f"Prediction   : {prediction}")
