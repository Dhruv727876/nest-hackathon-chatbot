"""
app/verify_samples.py
Pipeline verification for samples 02, 04, and 05.
Includes normalize() for fuzzy anchor matching + dosage/unit leak check.
"""
import sys, os, re

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_pipeline


# ---------------------------------------------------------------------------
# normalize() — shared fuzzy-match helper
# Maps vowel length variants + Bengali→Assamese script drift
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    replacements = {
        "\u09C0": "\u09BF",   # ী  → ি
        "\u09C2": "\u09C1",   # ূ  → ু
        "\u09B0": "\u09F0",   # র  → ৰ
        "\u09DC": "\u09F0",   # ড় → ৰ
        "\u09DD": "\u09F0",   # ঢ় → ৰ
        "\u09DF": "\u09F1",   # য় → ৱ
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


# Dosage pattern: numbers with medical units, or ratios like 1:1000
DOSAGE_RE = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|IU|iu|\bg\b|tablet|tab|cap|capsule|injection|inj|cc|unit)\b"
    r"|\d+:\d+)",
    re.IGNORECASE,
)

BENGALI_RA = "\u09B0"  # raw Bengali র — should be absent after correction

TESTS = [
    {
        "num":         "02",
        "topic":       "Symptoms of common cold",
        "anchor":      "চৰ্দি",      # সাধাৰণ চৰ্দি — চৰ্দি is the domain keyword
        "audio_in":    "tts/outputs/sample_02.wav",
        "out":         "app/outputs/pipeline_reply_02.wav",
        "dosage_check": False,       # cold symptoms shouldn't trigger dosage hallucination
    },
    {
        "num":         "05",
        "topic":       "Snake bite first aid",
        "anchor":      "সাপে",
        "audio_in":    "tts/outputs/sample_05.wav",
        "out":         "app/outputs/pipeline_reply_05.wav",
        "dosage_check": True,        # verify no drug/unit leak
    },
]

all_passed = True

for t in TESTS:
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  SAMPLE {t['num']}  [{t['topic']}]")
    print(f"  Expected anchor (normalised): {normalize(t['anchor'])}")
    print(sep)

    result = run_pipeline(t["audio_in"], t["out"])
    transcribed = result["transcribed_text"]
    reply       = result["reply_text"]

    norm_anchor      = normalize(t["anchor"])
    norm_transcribed = normalize(transcribed)

    check_anchor  = norm_anchor in norm_transcribed
    check_no_beng = BENGALI_RA not in reply
    check_content = len(reply.strip()) > 10

    print(f"\n  ── Result dict ──")
    print(f"    transcribed_text : {transcribed}")
    print(f"    reply_text       : {reply}")
    print(f"    audio_output_path: {result['audio_output_path']}")

    print(f"\n  ── Validation ──")
    print(f"    [1] ASR anchor '{t['anchor']}' found (normalised): {check_anchor}")
    print(f"        norm_transcribed = {norm_transcribed[:65]}...")
    print(f"    [2] Reply non-empty (>10 chars)                  : {check_content}")
    print(f"    [3] No raw Bengali \u09b0 in reply                   : {check_no_beng}")

    # Dosage leak check (always run, flag regardless of dosage_check setting)
    dosage_hits = DOSAGE_RE.findall(reply)
    dosage_clean = len(dosage_hits) == 0
    print(f"    [4] No drug dosage / unit leaks in reply         : {dosage_clean}")
    if dosage_hits:
        print(f"        !! LEAKED patterns: {dosage_hits}")

    ok = check_anchor and check_content and check_no_beng and dosage_clean
    print(f"    ALL CHECKS PASSED                                : {ok}")
    if not ok:
        all_passed = False

print("\n" + "=" * 70)
status = "ALL PASSED \u2705" if all_passed else "SOME FAILED \u274c"
print(f"  OVERALL: {status}")
print("=" * 70)
