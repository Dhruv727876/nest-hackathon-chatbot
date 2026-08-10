"""
app/batch_runner.py
Runs the full pipeline on all 15 samples sequentially.
Reports transcribed_text, reply_text, word count, and violations.
Outputs a markdown PASS/FAIL table of all 15 samples at the end.
"""
import sys, os, re

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import run_pipeline

# ── Import samples list from generate_samples ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts", "scripts"))
from generate_samples import SAMPLES

# ── Validation checks helpers ──────────────────────────────────────────────
WI_PREFIX_RE   = re.compile(r"(?<![^\s\(\[\-\,\.।])ৱি[ক-হ]\S+")
DEVANAGARI_RE  = re.compile(r"[\u0900-\u0963\u0966-\u097F]+")  # exclude danda U+0964, U+0965
LATIN_INLINE_RE= re.compile(r"[a-zA-Z]{4,}")
DOSAGE_RE      = re.compile(
    r"(?:\b\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|IU|iu|\bg\b|tablet|tab|cap|capsule|"
    r"injection|inj|cc|unit)\b|\d+:\d+)", re.IGNORECASE)

DRUG_NAMES_RE  = re.compile(
    r"(?:Paracetamol|Ibuprofen|Adrenaline|Dextrose|পেৰাচিটামল|আইবুপ্ৰফেন|এড্ৰেনালিন|ডেক্সট্ৰ’জ|মেটফৰ্মিন|Metformin)",
    re.IGNORECASE,
)

MD_RE = re.compile(r"\*|^\s*\d+\.\s+|^\s*\([ivxIVX]{1,4}\)[:\s]+|^\s*[-•]\s+", re.MULTILINE)

_FUNCTION_WORDS = frozenset({
    "আৰু", "বা", "যে", "লাগে", "হয়", "কৰা", "হলে", "হ'লে", "লৈ",
    "এই", "সেই", "তেওঁ", "আপুনি", "মই", "আমি", "তুমি", "কি", "কেনে",
    "কিয়", "কেতিয়া", "ক'ত", "কেনেকৈ", "আছে", "নাই", "পাৰে", "পাৰি",
    "দিয়া", "লওক", "যাওক", "কৰক", "থাকক", "থাকে", "হ'ব", "হ'ল",
})

BENGALI_RA = "\u09B0"

# Anchor definitions for all 15 samples (allowing list of synonyms)
ANCHORS = {
    1:  ["জ্বৰ"],
    2:  ["চৰ্দি", "সৰ্দি"],
    3:  ["চিকিৎসক", "ডাক্তৰ"],
    4:  ["টিকাকৰণ", "টীকাকৰণ"],
    5:  ["সাপে", "সাপ"],
    6:  ["মাতৃ", "গৰ্ভাৱস্থা", "বৰ্ভাৱস্থা"],
    7:  ["ডায়েবেটিচ", "ডায়েবেটিছ"],
    8:  ["ৰেচন", "ৰেশন"],
    9:  ["মাটি", "মাটিৰ"],
    10: ["অভিযোগ"],
    11: ["জন্ম", "জন্মৰ"],
    12: ["জলপানী", "জলপানি", "বৃত্তি"],
    13: ["বিদ্যুৎ", "বিজুলী"],
    14: ["পথ", "ৰাস্তা", "ৰাস্তাৰ"],
    15: ["ভোটাৰ"],
}

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

def repeated_content_words(text: str) -> list:
    from collections import Counter
    words = [re.sub(r"[^\u0980-\u09FF]", "", w) for w in text.split()]
    words = [w for w in words if len(w) >= 4 and w not in _FUNCTION_WORDS]
    return [w for w, c in Counter(words).items() if c >= 3]

results_table = []

for idx in range(1, 16):
    audio_path  = f"tts/outputs/sample_{idx:02d}.wav"
    output_path = f"app/outputs/pipeline_reply_{idx:02d}.wav"
    topic       = SAMPLES[idx - 1]["topic"]
    
    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  RUNNING PIPELINE: SAMPLE {idx:02d} — {topic}")
    print(sep)
    
    try:
        res = run_pipeline(audio_path, output_path)
        transcribed = res["transcribed_text"]
        reply       = res["reply_text"]
        wc          = len(reply.split())
        
        # Run Checks
        norm_trans  = normalize(transcribed)
        anchor_ok = False
        for a in ANCHORS[idx]:
            if normalize(a) in norm_trans:
                anchor_ok = True
                break
        no_beng_ok  = BENGALI_RA not in reply
        no_dosage   = len(DOSAGE_RE.findall(reply)) == 0
        no_drugs    = len(DRUG_NAMES_RE.findall(reply)) == 0
        no_md       = not bool(MD_RE.search(reply))
        no_dev      = len(DEVANAGARI_RE.findall(reply)) == 0
        no_wi       = len(WI_PREFIX_RE.findall(reply)) == 0
        no_reps     = len(repeated_content_words(reply)) == 0
        length_ok   = wc <= 90
        
        failures = []
        if not anchor_ok:  failures.append("ASR Anchor Mismatch")
        if not no_beng_ok: failures.append("Bengali Script Drift")
        if not no_dosage:  failures.append("Dosage Leaks")
        if not no_drugs:   failures.append("Drug Name Leak")
        if not no_md:      failures.append("Markdown/List Leak")
        if not no_dev:     failures.append("Devanagari Leak")
        if not no_wi:      failures.append("Nonsense word (ৱি-)")
        if not no_reps:    failures.append("Content Repetition")
        if not length_ok:  failures.append("Length > 90 words")
        
        status = "PASS" if not failures else "FAIL"
        details = ", ".join(failures) if failures else "All checks clean"
        
        results_table.append({
            "idx": idx,
            "topic": topic,
            "trans": transcribed,
            "reply": reply,
            "wc": wc,
            "status": status,
            "details": details
        })
        
        print(f"\n  ── SAMPLE {idx:02d} SUMMARY ──")
        print(f"    Transcribed: {transcribed}")
        print(f"    Reply      : {reply}")
        print(f"    Word Count : {wc}")
        print(f"    Status     : {status} ({details})")
        
    except Exception as e:
        results_table.append({
            "idx": idx,
            "topic": topic,
            "trans": "ERROR",
            "reply": str(e),
            "wc": 0,
            "status": "FAIL",
            "details": f"Pipeline execution error: {e}"
        })
        print(f"\n❌ Pipeline failed for Sample {idx:02d}: {e}")

# ── Print Final Markdown Table ──────────────────────────────────────────────
print("\n\n" + "=" * 75)
print("  FINAL BATCH RUNNER RESULTS TABLE")
print("=" * 75)
print("| Sample | Topic | Word Count | Status | Details |")
print("|--------|-------|------------|--------|---------|")
for r in results_table:
    print(f"| {r['idx']:02d} | {r['topic']} | {r['wc']} | {r['status']} | {r['details']} |")
print("=" * 75)
