"""
generate_samples.py — Batch Assamese TTS sample generation
===========================================================
Generates 15 WAV audio samples across healthcare and governance domains
using the text_to_speech() function from tts_function.py.

Output files : tts/outputs/sample_01.wav … sample_15.wav
Run from project root:
    asr\.venv\Scripts\python tts\scripts\generate_samples.py
"""

import sys
import os

# UTF-8 stdout for Assamese script output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))   # tts/scripts/
TTS_DIR     = os.path.dirname(SCRIPT_DIR)                  # tts/
OUTPUTS_DIR = os.path.join(TTS_DIR, "outputs")             # tts/outputs/

# Make tts_function importable regardless of CWD
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tts_function import text_to_speech  # noqa: E402

# ---------------------------------------------------------------------------
# 15 Assamese sample sentences — Healthcare (1–7) + Governance (8–15)
# ---------------------------------------------------------------------------
SAMPLES = [
    # ── Healthcare ──────────────────────────────────────────────────────────
    {
        "domain": "Healthcare",
        "topic":  "Fever and basic care advice",
        "text":   "জ্বৰ হ'লে কি কৰিব লাগে আৰু কেতিয়া চিকিৎসকৰ ওচৰলৈ যাব লাগে?",
    },
    {
        "domain": "Healthcare",
        "topic":  "Symptoms of common cold",
        "text":   "সাধাৰণ চৰ্দিৰ লক্ষণবোৰ কি কি আৰু ঘৰতে কেনেকৈ সুস্থ হ'ব পাৰি?",
    },
    {
        "domain": "Healthcare",
        "topic":  "How to book a doctor's appointment",
        "text":   "চিকিৎসকৰ সৈতে সাক্ষাতৰ সময় কেনেকৈ বুক কৰিব পাৰি?",
    },
    {
        "domain": "Healthcare",
        "topic":  "Vaccination schedule for children",
        "text":   "শিশুক কোনকোন বয়সত কোন কোন টীকা দিব লাগে?",
    },
    {
        "domain": "Healthcare",
        "topic":  "What to do in case of snake bite",
        "text":   "সাপে কামুৰিলে তৎক্ষণাৎ কি কৰিব লাগে?",
    },
    {
        "domain": "Healthcare",
        "topic":  "Maternal health checkup reminder",
        "text":   "গৰ্ভাৱস্থাত কিমান দিনৰ মূৰে মূৰে স্বাস্থ্য পৰীক্ষা কৰোৱা উচিত?",
    },
    {
        "domain": "Healthcare",
        "topic":  "Diabetes diet advice",
        "text":   "ডায়েবেটিছ ৰোগীয়ে কি কি খাদ্য খাব লাগে আৰু কি কি পৰিহাৰ কৰিব লাগে?",
    },

    # ── Governance ───────────────────────────────────────────────────────────
    {
        "domain": "Governance",
        "topic":  "How to apply for a ration card",
        "text":   "ৰেচন কাৰ্ডৰ বাবে আৱেদন কৰিবলৈ কি কি কাগজপত্ৰ লাগে আৰু ক'ত যাব লাগে?",
    },
    {
        "domain": "Governance",
        "topic":  "Checking land record status",
        "text":   "মোৰ মাটিৰ পট্টা আৰু ভূমি ৰেকৰ্ড অনলাইনত কেনেকৈ পৰীক্ষা কৰিব পাৰি?",
    },
    {
        "domain": "Governance",
        "topic":  "Filing a village-level grievance",
        "text":   "গাঁও পঞ্চায়ত পৰ্যায়ত কোনো সমস্যা থাকিলে ক'ত আৰু কেনেকৈ অভিযোগ দিব পাৰি?",
    },
    {
        "domain": "Governance",
        "topic":  "Applying for a birth certificate",
        "text":   "শিশুৰ জন্ম প্ৰমাণপত্ৰৰ বাবে আৱেদন কৰিবলৈ কি কৰিব লাগে?",
    },
    {
        "domain": "Governance",
        "topic":  "Information on government scholarship scheme",
        "text":   "অসম চৰকাৰৰ কোনকোন বৃত্তি আঁচনি উপলব্ধ আছে আৰু কেনেকৈ আৱেদন কৰিব পাৰি?",
    },
    {
        "domain": "Governance",
        "topic":  "How to pay electricity bill online",
        "text":   "বিদ্যুৎ বিল অনলাইনত কেনেকৈ পৰিশোধ কৰিব পাৰি?",
    },
    {
        "domain": "Governance",
        "topic":  "Reporting a road or infrastructure issue",
        "text":   "আমাৰ গাঁৱৰ ৰাস্তা ভাঙি গৈছে, এই সমস্যা চৰকাৰক কেনেকৈ জনাব পাৰি?",
    },
    {
        "domain": "Governance",
        "topic":  "Checking voter ID registration status",
        "text":   "মোৰ নাম ভোটাৰ তালিকাত আছে নে নাই কেনেকৈ জানিব পাৰি?",
    },
]

assert len(SAMPLES) == 15, f"Expected 15 samples, got {len(SAMPLES)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  Assamese TTS — Batch Sample Generation (15 samples)")
    print("  Model : facebook/mms-tts-asm")
    print("=" * 70)

    # Print the manifest upfront so the user can verify text before audio runs
    print("\n📋  Sample Manifest")
    print("-" * 70)
    for i, s in enumerate(SAMPLES, start=1):
        print(f"  [{i:02d}] [{s['domain']}] {s['topic']}")
        print(f"        {s['text']}")
    print("-" * 70)
    print()

    successes, failures = [], []

    for i, sample in enumerate(SAMPLES, start=1):
        output_file = os.path.join(OUTPUTS_DIR, f"sample_{i:02d}.wav")
        label = f"[{i:02d}/15] {sample['topic']}"

        print(f"🔊  Generating {label} …")
        try:
            text_to_speech(sample["text"], output_file)
            # tts_function already prints save details; add a blank line for readability
            print()
            successes.append(i)
        except Exception as e:
            print(f"     ❌  Failed: {e}\n", file=sys.stderr)
            failures.append((i, str(e)))

    # Final summary
    print("=" * 70)
    print(f"  Done — {len(successes)}/15 generated successfully.")
    if failures:
        print(f"\n  ⚠  Failed samples:")
        for idx, err in failures:
            print(f"     [{idx:02d}] {err}")
    print()
    print("  Output directory:", OUTPUTS_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
