"""
app/baseline_diagnostic.py
Raw baseline: confirm new model name + API params, then run pipeline on
samples 02 and 05 with zero code changes. Diagnostic only — no fixes.
"""
import sys, os, re

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

# ── Load env the same way main.py does ─────────────────────────────────────
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "nlu-mt", ".env")
load_dotenv(_env_path)

# ── 1. Confirm model + API params ─────────────────────────────────────────
MODEL_NAME        = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
FREQUENCY_PENALTY = 0.4
PRESENCE_PENALTY  = 0.3
MAX_TOKENS        = 300

print("=" * 70)
print("  BASELINE DIAGNOSTIC — New model check")
print("=" * 70)
print(f"  NIM_MODEL_NAME    : {MODEL_NAME}")
print(f"  frequency_penalty : {FREQUENCY_PENALTY}  (unchanged)")
print(f"  presence_penalty  : {PRESENCE_PENALTY}  (unchanged)")
print(f"  max_tokens        : {MAX_TOKENS}   (unchanged)")
print("=" * 70)

# ── Pattern detectors (analysis only — no truncation) ─────────────────────
WI_PREFIX_RE   = re.compile(r"\bৱি[ক-হ]\S+")          # ৱি-prefix nonsense
DEVANAGARI_RE  = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")  # exclude \u0964 \u0965 (danda, shared with Assamese)
LATIN_INLINE_RE= re.compile(r"[a-zA-Z]{4,}")           # multi-char Latin runs
DOSAGE_RE      = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|IU|iu|\bg\b|tablet|tab|cap|capsule|"
    r"injection|inj|cc|unit)\b|\d+:\d+)", re.IGNORECASE)

def word_count(text: str) -> int:
    return len(text.split())

def repeated_content_words(text: str, threshold: int = 3) -> list:
    """Return content words (4+ chars) that appear >= threshold times."""
    from collections import Counter
    words = re.findall(r"[\u0980-\u09FF]{4,}", text)  # Assamese words 4+ chars
    return [(w, c) for w, c in Counter(words).items() if c >= threshold]

def analyse(num: str, transcribed: str, reply: str):
    sep = "-" * 60
    wc  = word_count(reply)

    wi_hits     = WI_PREFIX_RE.findall(reply)
    dev_hits    = DEVANAGARI_RE.findall(reply)
    latin_hits  = LATIN_INLINE_RE.findall(reply)
    dosage_hits = DOSAGE_RE.findall(reply)
    rep_words   = repeated_content_words(reply)

    print(f"\n  ── Sample {num} Analysis ──")
    print(f"    transcribed_text : {transcribed}")
    print(f"    reply_text       : {reply}")
    print(f"    word_count       : {wc}")
    print(sep)
    print(f"  [A] ৱি-prefix fabricated words   : {wi_hits if wi_hits else 'None ✅'}")
    print(f"  [B] Devanagari script leaks       : {dev_hits[:5] if dev_hits else 'None ✅'}")
    print(f"  [C] Latin transliteration runs    : {latin_hits[:5] if latin_hits else 'None ✅'}")
    print(f"  [D] Drug dosage / unit patterns   : {dosage_hits if dosage_hits else 'None ✅'}")
    print(f"  [E] Content words repeated 3+×    : {rep_words if rep_words else 'None ✅'}")
    print(f"  [F] Word count <= 80              : {'✅' if wc <= 80 else f'❌ ({wc} words)'}")

# ── Run pipeline ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_pipeline

TESTS = [
    ("02", "Symptoms of common cold",    "tts/outputs/sample_02.wav", "app/outputs/baseline_reply_02.wav"),
    ("05", "Snake bite first aid",       "tts/outputs/sample_05.wav", "app/outputs/baseline_reply_05.wav"),
]

for num, topic, audio_in, audio_out in TESTS:
    print(f"\n{'='*70}")
    print(f"  SAMPLE {num}  [{topic}]")
    print(f"{'='*70}")
    result = run_pipeline(audio_in, audio_out)
    analyse(num, result["transcribed_text"], result["reply_text"])

print(f"\n{'='*70}")
print("  END OF BASELINE DIAGNOSTIC — no changes made to get_response.py")
print(f"{'='*70}")
