"""
tts_test.py — Assamese Text-to-Speech synthesis
================================================
Primary model  : facebook/mms-tts-asm  (Meta MMS · VITS · Assamese)
                 No gating, works directly via transformers VitsModel.
Fallback model : ai4bharat/indic-parler-tts  (Parler-TTS multilingual)
                 Requires `parler-tts` package; attempted if primary fails.

Output : tts/outputs/tts_sample.wav
"""

import sys
import os

# UTF-8 stdout so Assamese characters print correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths — resolved relative to THIS file so the script works from any CWD
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # tts/scripts/
TTS_DIR      = os.path.dirname(SCRIPT_DIR)                  # tts/
PROJECT_ROOT = os.path.dirname(TTS_DIR)                     # project root
OUTPUTS_DIR  = os.path.join(TTS_DIR, "outputs")             # tts/outputs/
OUTPUT_PATH  = os.path.join(OUTPUTS_DIR, "tts_sample.wav")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Text to synthesise
# ---------------------------------------------------------------------------
TEXT = "নমস্কাৰ, আপুনি কেনে আছে?"

# ---------------------------------------------------------------------------
# Helper — save numpy/torch audio to WAV
# ---------------------------------------------------------------------------
def save_wav(audio, sample_rate: int, path: str):
    import numpy as np
    import soundfile as sf

    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    audio = audio.squeeze()                 # remove batch / channel dims
    audio = audio.astype("float32")

    # Normalise to [-1, 1] to avoid clipping
    peak = float(abs(audio).max())
    if peak > 0:
        audio = audio / peak

    sf.write(path, audio, sample_rate)
    print(f"\n✅ Audio saved → {path}")
    print(f"   Sample rate  : {sample_rate} Hz")
    print(f"   Duration     : {len(audio)/sample_rate:.2f} s")
    print(f"   Samples      : {len(audio)}")


# ---------------------------------------------------------------------------
# Strategy 1 — facebook/mms-tts-asm  (recommended, no gating)
# ---------------------------------------------------------------------------
def run_mms_tts() -> bool:
    MODEL_ID = "facebook/mms-tts-asm"
    print(f"\n{'='*60}")
    print(f"  Model : {MODEL_ID}")
    print(f"  Type  : Meta MMS · VITS · Assamese-specific")
    print(f"{'='*60}")

    try:
        from transformers import VitsModel, AutoTokenizer
        import torch
    except ImportError as e:
        print(f"[MMS] Missing dependency: {e}")
        return False

    try:
        print(f"\n[MMS] Loading tokeniser …")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        print(f"[MMS] Loading model …")
        model = VitsModel.from_pretrained(MODEL_ID)
        model.eval()

        print(f"\n[MMS] Input text : {TEXT}")
        inputs = tokenizer(TEXT, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs)

        # VitsModel returns a ModelOutput; waveform is in .waveform
        waveform = output.waveform          # shape: (1, num_samples)
        sr = model.config.sampling_rate     # typically 16 000 Hz

        save_wav(waveform, sr, OUTPUT_PATH)
        print(f"\n  Checkpoint used: {MODEL_ID}")
        return True

    except Exception as e:
        print(f"[MMS] Error: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Strategy 2 — ai4bharat/indic-parler-tts  (fallback)
# ---------------------------------------------------------------------------
def run_indic_parler_tts() -> bool:
    MODEL_ID = "ai4bharat/indic-parler-tts"
    print(f"\n{'='*60}")
    print(f"  Model : {MODEL_ID}")
    print(f"  Type  : AI4Bharat Indic Parler-TTS · multilingual")
    print(f"{'='*60}")

    # parler-tts is not in the base requirements — install on-the-fly
    try:
        import parler_tts  # noqa: F401
    except ImportError:
        print("[Parler] parler-tts not found — installing …")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/huggingface/parler-tts.git"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"[Parler] Install failed:\n{result.stderr}", file=sys.stderr)
            return False

    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import torch
    except ImportError as e:
        print(f"[Parler] Import failed after install: {e}", file=sys.stderr)
        return False

    # Retrieve HF token (needed if model is gated)
    token = _get_hf_token()

    try:
        print("[Parler] Loading tokeniser and model (may take a while) …")
        tokenizer  = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        model      = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID, token=token)
        model.eval()

        # Parler-TTS requires a natural-language voice description
        DESCRIPTION = (
            "A female speaker delivers a very expressive Assamese sentence "
            "with a clear, natural voice at a moderate pace."
        )

        print(f"\n[Parler] Input text  : {TEXT}")
        print(f"[Parler] Voice prompt: {DESCRIPTION}")

        desc_inputs = tokenizer(DESCRIPTION, return_tensors="pt")
        text_inputs = tokenizer(TEXT,        return_tensors="pt")

        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                prompt_input_ids=text_inputs.input_ids,
            )

        sr = model.config.sampling_rate
        save_wav(generation, sr, OUTPUT_PATH)
        print(f"\n  Checkpoint used: {MODEL_ID}")
        return True

    except Exception as e:
        print(f"[Parler] Error: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Helper — load HF_TOKEN from env or .env files
# ---------------------------------------------------------------------------
def _get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    candidates = [
        os.path.join(PROJECT_ROOT, ".env"),
        os.path.join(PROJECT_ROOT, "nlu-mt", ".env"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN="):
                            val = line.split("=", 1)[1].strip().strip("\"'")
                            return val or None
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print(f"\nAssamese TTS — synthesising speech …")
    print(f"Text   : {TEXT}")
    print(f"Output : {OUTPUT_PATH}\n")

    # Try primary model first
    if run_mms_tts():
        return

    print("\n[INFO] Primary model failed — trying fallback (Indic Parler-TTS) …")
    if run_indic_parler_tts():
        return

    print(
        "\n❌ Both TTS models failed. Possible fixes:\n"
        "   1. Check your internet connection.\n"
        "   2. Ensure transformers >= 4.33 is installed.\n"
        "   3. For Indic Parler-TTS, make sure HF_TOKEN is set in nlu-mt/.env.\n",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
