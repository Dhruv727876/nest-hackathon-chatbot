"""
app/main.py — Full ASR → NLU → TTS Pipeline
=============================================
Wires three modules together:
  1. speech_to_text()  — asr/scripts/asr_function.py  (IndicConformer)
  2. get_response()    — nlu-mt/get_response.py        (NVIDIA NIM / LLaMA)
  3. text_to_speech()  — tts/scripts/tts_function.py  (Meta MMS-TTS Assamese)

Usage (from project root):
    asr\\.venv\\Scripts\\python app/main.py [optional_audio_path]
"""

import sys
import os

# ---------------------------------------------------------------------------
# UTF-8 stdout for Assamese script on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths — all resolved relative to THIS file so the script is CWD-agnostic
# ---------------------------------------------------------------------------
APP_DIR      = os.path.dirname(os.path.abspath(__file__))   # app/
PROJECT_ROOT = os.path.dirname(APP_DIR)                     # project root

ASR_SCRIPTS  = os.path.join(PROJECT_ROOT, "asr",    "scripts")
NLU_MT_DIR   = os.path.join(PROJECT_ROOT, "nlu-mt")
TTS_SCRIPTS  = os.path.join(PROJECT_ROOT, "tts",    "scripts")
APP_OUTPUTS  = os.path.join(APP_DIR, "outputs")

# Add sub-module directories to sys.path so their imports resolve correctly
for _p in [ASR_SCRIPTS, NLU_MT_DIR, TTS_SCRIPTS, PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.makedirs(APP_OUTPUTS, exist_ok=True)

# ---------------------------------------------------------------------------
# Load environment variables from nlu-mt/.env explicitly so NVIDIA_API_KEY
# is available regardless of which directory the script is run from.
# (get_response.py calls load_dotenv() without a path — defaults to CWD)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(NLU_MT_DIR, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=True)
except ImportError:
    pass  # python-dotenv not installed; env vars must be set externally

# ---------------------------------------------------------------------------
# Import the three pipeline components
# ---------------------------------------------------------------------------
try:
    from asr_function import speech_to_text
except ImportError as e:
    raise ImportError(
        f"Could not import speech_to_text from asr/scripts/asr_function.py: {e}\n"
        "Make sure asr/scripts/asr_function.py exists and dependencies are installed."
    )

try:
    from get_response import get_response
except ImportError as e:
    raise ImportError(
        f"Could not import get_response from nlu-mt/get_response.py: {e}\n"
        "Make sure nlu-mt/get_response.py exists and python-dotenv + requests are installed."
    )

try:
    from tts_function import text_to_speech
except ImportError as e:
    raise ImportError(
        f"Could not import text_to_speech from tts/scripts/tts_function.py: {e}\n"
        "Make sure tts/scripts/tts_function.py exists and transformers/soundfile are installed."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(audio_path: str, output_path: str) -> dict:
    """
    Run the full spoken-dialogue pipeline on a single audio file.

    Parameters
    ----------
    audio_path : str
        Path to the input WAV/FLAC file containing Assamese speech.
        Also accepts a FLEURS-style audio dict with a 'bytes' key.
    output_path : str
        Path where the synthesised reply WAV will be saved.
        Parent directories are created automatically.

    Returns
    -------
    dict with keys:
        "transcribed_text"  — Assamese text recognised from audio
        "reply_text"        — Assamese LLM response to that text
        "audio_output_path" — Absolute path to the synthesised reply WAV
    """
    sep = "─" * 60

    # ── Step 1: ASR ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 1 / 3 — Speech → Text  (IndicConformer)")
    print(sep)
    transcribed_text = speech_to_text(audio_path)
    print(f"  Transcribed: {transcribed_text}")

    # ── Step 2: NLU / LLM ───────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 2 / 3 — Text → Response  (NVIDIA NIM LLaMA)")
    print(sep)
    reply_text = get_response(transcribed_text)
    print(f"  Reply: {reply_text}")

    # ── Step 3: TTS ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Step 3 / 3 — Response → Speech  (MMS-TTS Assamese)")
    print(sep)
    text_to_speech(reply_text, output_path)

    abs_output = os.path.abspath(output_path)

    result = {
        "transcribed_text":  transcribed_text,
        "reply_text":        reply_text,
        "audio_output_path": abs_output,
    }
    return result


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full Assamese spoken-dialogue pipeline."
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default=None,
        help="Path to input WAV file (default: first FLEURS train sample).",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(APP_OUTPUTS, "pipeline_reply.wav"),
        help="Output WAV path for the synthesised reply.",
    )
    args = parser.parse_args()

    # ── Determine test audio input ───────────────────────────────
    if args.audio_path:
        if not os.path.exists(args.audio_path):
            print(f"Error: Audio file not found: {args.audio_path}", file=sys.stderr)
            sys.exit(1)
        test_input = args.audio_path
        print(f"\n🎙  Input audio : {test_input}")
    else:
        # Fall back to loading first FLEURS sample (cached locally)
        print("\n🎙  No audio file provided — loading first FLEURS 'as_in' train sample …")
        try:
            import datasets as hf_datasets
            ds = hf_datasets.load_dataset("google/fleurs", "as_in", split="train")
            ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))
            test_input = ds[0]["audio"]  # dict with 'bytes' key — accepted by speech_to_text()
            print(f"   Ground truth: {ds[0].get('transcription', '')}")
        except Exception as e:
            print(f"Error loading FLEURS dataset: {e}", file=sys.stderr)
            print(
                "Tip: Provide a WAV file as an argument:\n"
                "  asr\\.venv\\Scripts\\python app/main.py path/to/audio.wav",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"💾  Output WAV  : {args.output}\n")

    # ── Run the pipeline ─────────────────────────────────────────
    try:
        result = run_pipeline(test_input, args.output)
    except Exception as e:
        print(f"\n❌  Pipeline failed: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    # ── Print result dict ────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ✅  Pipeline complete — result:")
    print("═" * 60)
    for key, val in result.items():
        print(f"  {key}:")
        print(f"      {val}")
    print("═" * 60)
