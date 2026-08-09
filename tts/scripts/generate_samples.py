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
        "text":   (
            "জ্বৰ হ'লে বেছি পানী খাওক, জিৰণি লওক, "
            "আৰু প্ৰয়োজন হ'লে চিকিৎসকৰ পৰামৰ্শ লওক।"
        ),
    },
    {
        "domain": "Healthcare",
        "topic":  "Symptoms of common cold",
        "text":   (
            "সাধাৰণ চৰ্দিৰ লক্ষণসমূহ হৈছে নাক বন্ধ হোৱা, "
            "হাঁচি মাৰা, গলা বিষ আৰু লঘু জ্বৰ।"
        ),
    },
    {
        "domain": "Healthcare",
        "topic":  "How to book a doctor's appointment",
        "text":   (
            "চিকিৎসকৰ সৈতে পৰামৰ্শৰ বাবে "
            "আপোনাৰ নিকটৱৰ্তী প্ৰাথমিক স্বাস্থ্যকেন্দ্ৰত "
            "ফোন কৰক বা পোনে পোনে গৈ সময় বুক কৰক।"
        ),
    },
    {
        "domain": "Healthcare",
        "topic":  "Vaccination schedule for children",
        "text":   (
            "শিশুৰ সুৰক্ষাৰ বাবে জন্মৰ পিছৰে পৰা "
            "নিৰ্ধাৰিত সময়সূচী অনুযায়ী সকলো টিকা দিয়াটো অত্যন্ত জৰুৰী।"
        ),
    },
    {
        "domain": "Healthcare",
        "topic":  "What to do in case of snake bite",
        "text":   (
            "সাপে কামুৰিলে তৎক্ষণাৎ কামোৰা ঠাইখন স্থিৰ ৰাখক, "
            "ক্ষতস্থান কাটিব নালাগে, "
            "আৰু দ্ৰুততাৰে নিকটৱৰ্তী চিকিৎসালয়ত যোগাযোগ কৰক।"
        ),
    },
    {
        "domain": "Healthcare",
        "topic":  "Maternal health checkup reminder",
        "text":   (
            "গৰ্ভাৱস্থাত নিয়মিত স্বাস্থ্য পৰীক্ষা কৰোৱাটো "
            "মাতৃ আৰু শিশু উভয়ৰে সুস্বাস্থ্যৰ বাবে অপৰিহাৰ্য।"
        ),
    },
    {
        "domain": "Healthcare",
        "topic":  "Diabetes diet advice",
        "text":   (
            "ডায়েবেটিছ ৰোগীসকলে মিঠা আৰু শৰ্কৰাযুক্ত খাদ্য পৰিহাৰ কৰি "
            "সুষম আহাৰ গ্ৰহণ কৰক আৰু নিয়মিতভাৱে তেজৰ শৰ্কৰা পৰীক্ষা কৰক।"
        ),
    },

    # ── Governance ───────────────────────────────────────────────────────────
    {
        "domain": "Governance",
        "topic":  "How to apply for a ration card",
        "text":   (
            "ৰেচন কাৰ্ডৰ বাবে আৱেদন কৰিবলৈ "
            "আপোনাৰ ওচৰৰ খাদ্য আৰু অসামৰিক যোগান কাৰ্যালয়ত "
            "প্ৰয়োজনীয় কাগজপত্ৰ লৈ যোগাযোগ কৰক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "Checking land record status",
        "text":   (
            "আপোনাৰ মাটিৰ কাগজ আৰু ভূমি ৰেকৰ্ড পৰীক্ষা কৰিবলৈ "
            "অসম চৰকাৰৰ ধৰিত্ৰী অনলাইন প'ৰ্টেলত লগ ইন কৰক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "Filing a village-level grievance",
        "text":   (
            "গাঁও পঞ্চায়ত পৰ্যায়ত কোনো সমস্যা বা অভিযোগ দাখিল কৰিবলৈ "
            "আপোনাৰ গাঁওবুঢ়া বা পঞ্চায়ত সচিৱৰ সৈতে যোগাযোগ কৰক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "Applying for a birth certificate",
        "text":   (
            "জন্ম প্ৰমাণপত্ৰৰ বাবে শিশুৰ জন্মৰ একুৰি এদিনৰ ভিতৰত "
            "নিকটৱৰ্তী পৌৰসভা বা গ্ৰাম পঞ্চায়তত আৱেদন কৰক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "Information on government scholarship scheme",
        "text":   (
            "অসম চৰকাৰৰ বৃত্তি আঁচনিৰ বিষয়ে বিস্তাৰিত তথ্য পাবলৈ "
            "ই-ডিষ্ট্ৰিক্ট অসম প'ৰ্টেল বা আপোনাৰ বিদ্যালয়ৰ প্ৰধান শিক্ষকৰ সৈতে কথা পাতক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "How to pay electricity bill online",
        "text":   (
            "বিদ্যুৎ বিল অনলাইনত পৰিশোধ কৰিবলৈ "
            "APDCL-ৰ অফিচিয়েল ৱেবছাইট বা মোবাইল এপ ব্যৱহাৰ কৰক "
            "আৰু আপোনাৰ গ্ৰাহক নম্বৰ দিয়ক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "Reporting a road or infrastructure issue",
        "text":   (
            "ৰাস্তাৰ গাঁত বা আন্তঃগাঁথনিৰ কোনো সমস্যা জনাবলৈ "
            "অসম চৰকাৰৰ অনলাইন অভিযোগ প'ৰ্টেলত অভিযোগ দাখিল কৰক।"
        ),
    },
    {
        "domain": "Governance",
        "topic":  "Checking voter ID registration status",
        "text":   (
            "আপোনাৰ ভোটাৰ পৰিচয়পত্ৰৰ তথ্য পৰীক্ষা কৰিবলৈ "
            "ভাৰতীয় নিৰ্বাচন আয়োগৰ ৱেবছাইটত গৈ "
            "আপোনাৰ নাম আৰু ঠিকনা দি সন্ধান কৰক।"
        ),
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
