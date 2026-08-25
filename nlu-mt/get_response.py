import os
import sys
import re
import requests
from dotenv import load_dotenv

# Load environment variables from a .env file if available
load_dotenv()

try:
    from nlu_mt.assamese_examples import ASSAMESE_EXAMPLES
except ImportError:
    from assamese_examples import ASSAMESE_EXAMPLES

# Configurable model name for NVIDIA NIM API
# Alternative options to try:
# - "meta/llama-3.1-8b-instruct" (default, confirmed ~1s latency)
# - "google/gemma-2-9b-it" (fallback)
# Can be overridden via NIM_MODEL_NAME environment variable
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")

# Demo Mode Safe Fallback Mode flag (Fix #2)
DEMO_MODE = False

# Globals for test runner inspection
last_raw_reply = None
last_qc_triggered = False


# ---------------------------------------------------------------------------
# Normalization Helper
# Maps both vowel-length variants (ী→ি, ূ→ু) and script-drift characters (র/ড়/ঢ়→ৰ, য়→ৱ)
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Normalize Assamese text to make matching robust to script drift and spelling variations."""
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


# ---------------------------------------------------------------------------
# TEMPLATES dict (Template-first routing)
# Maps emergency and standard demo topics using normalized keyword lists.
# Order is preserved: emergency templates are checked first.
# ---------------------------------------------------------------------------
TEMPLATES = {
    # ── Emergency / Safety Critical Templates (Checked first) ──────────────────
    "emergency_infant_fever": {
        "keywords": [("শিশু", "জ্বৰ"), ("নৱজাতক", "জ্বৰ")],
        "answer": "নৱজাতক বা শিশুৰ জ্বৰ হ'লে পলম নকৰি তৎক্ষণাৎ চিকিৎসকৰ ওচৰলৈ লৈ যাওক। ঘৰুৱাভাৱে কোনো ঔষধ নিদিব আৰু চিকিৎসকৰ পৰামৰ্শ লওক।"
    },
    "emergency_dosage_query": {
        "keywords": ["ডোজ", "মিলিগ্ৰাম", "কিমান ঔষধ", "কিমান বটিকা"],
        "answer": "ঔষধৰ সঠিক পৰিমাণ আৰু ডোজৰ বাবে অনুগ্ৰহ কৰি চিকিৎসক বা ফাৰ্মাচিষ্টৰ পৰামৰ্শ লওক। নিজে ডোজ নিৰ্ধাৰণ কৰিব নালাগে।"
    },
    "emergency_poisoning": {
        "keywords": ["বিহ", "বিষক্ৰia", "জহৰ"],
        "answer": "বিষক্ৰিয়াৰ সন্দেহ হ'লে তৎক্ষণাৎ নিকটৱৰ্তী চিকিৎসালয়লৈ যাওক। চিকিৎসকৰ পৰামৰ্শ নোলোৱাকৈ বমি কৰাবলৈ চেষ্টা নকৰিব।"
    },
    "emergency_bleeding_accident": {
        "keywords": ["তেজ", "দুৰ্ঘটনা", "ৰক্তক্ষৰণ"],
        "answer": "দুৰ্ঘটনা বা গুৰুতৰ ৰক্তক্ষৰণ হ'লে লগে লগে নিকটৱৰ্তী চিকিৎসালয়লৈ লৈ যাওক। ক্ষতস্থানত চাফা কাপোৰেৰে হেঁচি ধৰি ৰক্তক্ষৰণ বন্ধ কৰিবলৈ চেষ্টা কৰক।"
    },

    # ── The 15 Standard Demo Topics (Checked second) ─────────────────────────
    "topic_01_fever": {
        "keywords": ["জ্বৰ"],
        "answer": "জ্বৰ হ'লে বেছি পানী খাওক, সম্পূৰ্ণ জিৰণি লওক আৰু গা ঠাণ্ডা কাপোৰেৰে মচি থাকক। জ্বৰ ৩ দিনতকৈ বেছি থাকিলে বা অত্যধিক বাঢ়িলে তৎক্ষণাৎ চিকিৎসকৰ পৰামৰ্শ লওক।"
    },
    "topic_02_common_cold": {
        "keywords": ["চৰ্দি", "সৰ্দি"],
        "answer": "সাধাৰণ চৰ্দিৰ লক্ষণ হ'ল নাক বন্ধ হোৱা, হাঁচি মৰা, গলা বিষ আৰু লঘু জ্বৰ। গৰম পানী খাই আৰু পৰ্যাপ্ত জিৰণি লৈ আৰাম পাব পাৰি, লক্ষণ নাইকিয়া নহ'লে চিকিৎসকৰ পৰামৰ্শ লওক।"
    },
    "topic_03_doctor_appointment": {
        "keywords": [
            ("চিকিৎসক", "সাক্ষাত"), ("ডাক্তৰ", "সাক্ষাত"),
            ("চিকিৎসক", "সময়"), ("ডাক্তৰ", "সময়"),
            ("চিকিৎসক", "বুক"), ("ডাক্তৰ", "বুক")
        ],
        "answer": "চিকিৎসকৰ সৈতে সাক্ষাতৰ সময় বুক কৰিবলৈ আপোনাৰ নিকটৱৰ্তী স্বাস্থ্যকেন্দ্ৰ বা চিকিৎসালয়ত ফোন কৰক বা পোনে পোনে গৈ পঞ্জীয়ন কৰক।"
    },
    "topic_04_vaccination": {
        "keywords": [
            "টিকাকৰণ", "টীকাকৰণ",
            ("টিকা", "সময়সূচী"), ("টীকা", "সময়সূচী"),
            ("টিকা", "শিশু"), ("টীকা", "শিশু")
        ],
        "answer": "শিশুৰ বাবে জন্মৰ পিছৰ পৰাই নিৰ্ধাৰিত সময়সূচী অনুযায়ী সকলো টিকা দিয়াটো অত্যন্ত জৰুৰী। সঠিক সময়সূচীৰ বাবে আপোনাৰ নিকটৱৰ্তী স্বাস্থ্যকেন্দ্ৰত যোগাযোগ কৰক।"
    },
    "topic_05_snake_bite": {
        "keywords": [("সাপে", "কামুৰ"), ("সাপ", "কামুৰ"), ("সাপে", "দংশন"), "সৰীসৃপ"],
        "answer": "সাপে কামুৰিলে এই পদক্ষেপবোৰ লওক: ৰোগীক শান্ত আৰু স্থিৰ ৰাখক। কামোৰা অংগটো হৃদয়ৰ তলত ৰাখক আৰু নুমুৱাব। ক্ষতস্থান কাটিব নালাগে বা মুখেৰে বিষ চুহিব নালাগে। তৎক্ষণাৎ নিকটৱৰ্তী চিকিৎসালয়লৈ লৈ যাওক।"
    },
    "topic_06_maternal_health": {
        "keywords": ["গৰ্ভাৱস্থা", "বৰ্ভাৱস্থা", ("মাতৃ", "স্বাস্থ্য"), ("মাতৃ", "পৰীক্ষা")],
        "answer": "গৰ্ভাৱস্থাত নিয়মিত স্বাস্থ্য পৰীক্ষা কৰোৱাটো মাতৃ আৰু শিশু উভয়ৰে সুস্বাস্থ্যৰ বাবে অপৰিহাৰ্য। নিকটৱৰ্তী স্বাস্থ্যকেন্দ্ৰত নিয়মীয়াকৈ পৰীক্ষা কৰোৱাই থাকক।"
    },
    "topic_07_diabetes_diet": {
        "keywords": ["ডায়েবেটিচ", "ডায়েবেটিছ"],
        "answer": "ডায়েবেটিছ ৰোগীয়ে মিঠা আৰু শৰ্কৰাযুক্ত খাদ্য পৰিহাৰ কৰি সুষম আহাৰ গ্ৰহণ কৰক আৰু নিয়মিতভাৱে তেজৰ শৰ্কৰা পৰীক্ষা কৰক। খাদ্যতালিকা প্ৰস্তুতিৰ বাবে চিকিৎসকৰ পৰামৰ্শ লওক।"
    },
    "topic_08_ration_card": {
        "keywords": ["ৰেচন", "ৰেশন"],
        "answer": "ৰেচন কাৰ্ডৰ বাবে আৱেদন কৰিবলৈ আপোনাৰ ওচৰৰ খাদ্য আৰু অসামৰিক যোগান কাৰ্যালয়ত নাগৰিকত্বৰ প্ৰমাণ আৰু ঠিকনাৰ প্ৰমাণসহ প্ৰয়োজনীয় কাগজপত্ৰ লৈ যোগাযোগ কৰক।"
    },
    "topic_09_land_record": {
        "keywords": [
            ("মাটি", "ৰেকৰ্ড"), ("মাটিৰ", "ৰেকৰ্ড"),
            ("মাটি", "দলিল"), ("মাটিৰ", "দলিল"),
            ("মাটি", "কাগজ"), ("মাটিৰ", "কাগজ"),
            "ধৰিত্ৰী", "ৰেক্ডো",
            ("মাটি", "অনলাইন"), ("মাটিৰ", "অনলাইন")
        ],
        "answer": "আপোনাৰ মাটিৰ কাগজ আৰু ভূমি ৰেকৰ্ড পৰীক্ষা কৰিবলৈ অসম চৰকাৰৰ ধৰিত্ৰী অনলাইন প'ৰ্টেলত লগ ইন কৰক, বা নিকটৱৰ্তী সাৰ্কল কাৰ্যালয়ত যোগাযোগ কৰক।"
    },
    "topic_11_birth_certificate": {
        "keywords": [
            ("জন্ম", "প্ৰমাণপত্ৰ"), ("জন্মৰ", "প্ৰমাণপত্ৰ"),
            ("জন্ম", "প্ৰমাণ"), ("জন্মৰ", "প্ৰমাণ"),
            ("জন্ম", "আৱেদন"), ("জন্মৰ", "আৱেদন"),
            ("জন্ম", "ৰেকৰ্ড"), ("জন্ম", "পঞ্জীয়ন")
        ],
        "answer": "জন্ম প্ৰমাণপত্ৰৰ বাবে শিশুৰ জন্মৰ পিছত যিমান সোনকালে সম্ভৱ নিকটৱৰ্তী পৌৰসভা বা গাঁও পঞ্চayতত আৱেদন কৰক।"
    },
    "topic_12_scholarship": {
        "keywords": [
            "জলpানী", "জলপানী", "জলপানি",
            ("বৃত্তি", "আঁচনি"), ("বৃত্তি", "আৱেদন"), ("বৃত্তি", "চৰকাৰী")
        ],
        "answer": "চৰকাৰী বৃত্তি বা জলপানী আঁচনিৰ বিষয়ে বিস্তাৰিত তথ্য পাবলৈ ই-ডিষ্ট্ৰিক্ট অসম প'ৰ্টেল চাওক, বা আপোনাৰ বিদ্যালয়ৰ প্ৰধান শিক্ষকৰ সৈতে কথা পাতক।"
    },
    "topic_13_electricity_bill": {
        "keywords": [
            ("বিদ্যুৎ", "বিল"), ("বিজুলী", "বিল"),
            ("বিদ্যুৎ", "পৰিশোধ"), ("বিজুলী", "পৰিশোধ"),
            ("বিদ্যুৎ", "সংযোগ"), ("বিজুলী", "সংযোগ"),
            "APDCL"
        ],
        "answer": "বিদ্যুৎ বিল অনলাইনত পৰিশোধ কৰিবলৈ APDCL-ৰ অফিচিয়েল ৱেবছাইট বা মোবাইল এপ ব্যৱহাৰ কৰক আৰু আপোনাৰ গ্ৰাহক নম্বৰ দিয়ক।"
    },
    "topic_14_road_infrastructure": {
        "keywords": [
            ("ৰাস্তা", "গাঁত"), ("ৰাস্তা", "আন্তঃগাঁথনি"), ("ৰাস্তা", "সমস্যা"),
            ("ৰাস্তাৰ", "গাঁত"), ("ৰাস্তাৰ", "আন্তঃগাঁথনি"), ("ৰাস্তাৰ", "সমস্যা"),
            ("পথ", "গাঁত"), ("পথ", "আন্তঃগাঁথনি"), ("পথ", "সমস্যা")
        ],
        "answer": "ৰাস্তাৰ গাঁত বা আন্তঃগাঁথনিৰ কোনো সমস্যা জনাবলৈ অসম চৰকাৰৰ অনলাইন অভিযোগ প'ৰ্টেলত অভিযোগ দাখিল কৰক, বা স্থানীয় পঞ্চায়ত কাৰ্যালয়ত জনাওক।"
    },
    "topic_10_village_grievance": {
        "keywords": [
            ("অভিযোগ", "গাঁও"), ("অভিযোগ", "পঞ্চায়ত"),
            "গাঁও পঞ্চায়ত"
        ],
        "answer": "গাঁও পঞ্চায়ত পৰ্যায়ত কোনো সমস্যা বা অভিযোগ দাখিল কৰিবলৈ আপোনাৰ গাঁওবুঢ়া বা পঞ্চায়ত সচিৱৰ সৈতে যোগাযোগ কৰক।"
    },
    "topic_15_voter_id": {
        "keywords": ["ভোটাৰ"],
        "answer": "আপোনাৰ ভোটাৰ পঞ্জীয়নৰ তথ্য পৰীক্ষা কৰিবলৈ ৰাষ্ট্ৰীয় ভোটাৰ সেৱা পৰ্টেল (NVSP)-ৰ ৱেবছাইটত গৈ নিজৰ নাম আৰু ঠিকনা দি সন্ধান কৰক।"
    },
}


# ---------------------------------------------------------------------------
# Helper constants for post-processing (Used in LLM Fallback Path)
# ---------------------------------------------------------------------------

# Common Assamese function words excluded from unigram repetition guard
_FUNCTION_WORDS = frozenset({
    "আৰু", "বা", "যে", "লাগে", "হয়", "কৰা", "হলে", "হ'লে", "লৈ",
    "এই", "সেই", "তেওঁ", "আপুনি", "মই", "আমি", "তুমি", "কি", "কেনে",
    "কিয়", "কেতিয়া", "ক'ত", "কেনেকৈ", "আছে", "নাই", "পাৰে", "পাৰি",
    "দia", "লওক", "যাওক", "কৰক", "থাকক", "থাকে", "হ'в", "হ'ল",
})

# Markdown / list pattern cleaner
_MD_PATTERN = re.compile(
    r"\*{1,3}[^*\n]*\*{1,3}"        # **bold** or *italic*
    r"|^\s*\d+\.\s+"                 # 1. 2. 3. list markers
    r"|^\s*\([ivxIVX]{1,4}\)[:\s]+" # (i): (ii): style markers
    r"|^\s*[-•]\s+",                 # bullet points
    re.MULTILINE,
)

# Dosage pattern
_DOSAGE_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|IU|iu|g\b|tablet|tab|cap|capsule|injection|inj|cc|unit)\b"
    r"|\d+:\d+"    # ratio patterns like 1:1000
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# LLM Post-processing Engine
# ---------------------------------------------------------------------------

def _postprocess(reply: str) -> str:
    """Apply all post-processing guards in sequence and return cleaned reply."""

    # ── (4) Strip markdown and list formatting ─────────────────────────────
    reply = _MD_PATTERN.sub("", reply)
    reply = re.sub(r"\n{2,}", " ", reply)   # collapse blank lines
    reply = re.sub(r"\s{2,}", " ", reply).strip()

    # ── (3a) 3-gram repetition guard ──────────────────────────────────────
    words = reply.split()
    if len(words) >= 3:
        seen_trigrams: dict = {}
        repeated_phrase = None
        repeated_start_idx = None
        for idx in range(len(words) - 2):
            trigram = (words[idx], words[idx + 1], words[idx + 2])
            if trigram in seen_trigrams:
                if repeated_phrase is None:
                    repeated_phrase = trigram
                    repeated_start_idx = seen_trigrams[trigram]
            else:
                seen_trigrams[trigram] = idx
        if repeated_phrase is not None:
            # Truncate at the start of the second occurrence (idx) instead of the first,
            # ensuring the first occurrence and preceding words are preserved.
            # Only truncate if we have at least 3 words remaining.
            cutoff_word_idx = max(3, idx)
            reply = " ".join(words[:cutoff_word_idx]).rstrip(",;- ") + "।"
            print(
                f"[get_response] \u26a0 3-gram repetition detected "
                f"('{' '.join(repeated_phrase)}') \u2014 truncated reply.",
                file=sys.stderr,
            )

    # ── (3b) Unigram repetition guard (content words 4+ chars) ───────────
    words = reply.split()
    word_counts: dict = {}
    for w in words:
        clean = re.sub(r"[^\u0980-\u09FF]", "", w)   # strip non-Assamese chars
        if len(clean) >= 4 and clean not in _FUNCTION_WORDS:
            word_counts[clean] = word_counts.get(clean, 0) + 1
    repeated_unigrams = {w: c for w, c in word_counts.items() if c >= 3}
    if repeated_unigrams:
        offender = next(iter(repeated_unigrams))
        count = 0
        cutoff_idx = len(words)
        for idx, w in enumerate(words):
            clean = re.sub(r"[^\u0980-\u09FF]", "", w)
            if clean == offender:
                count += 1
                if count == 3:   # truncate BEFORE 3rd occurrence
                    cutoff_idx = idx
                    break
        reply = " ".join(words[:cutoff_idx]).rstrip(",;- ") + "।"
        print(
            f"[get_response] \u26a0 Unigram repetition ('{offender}' "
            f"\xd7{repeated_unigrams[offender]}) \u2014 truncated before 3rd occurrence.",
            file=sys.stderr,
        )

    # ── (2) Word-count hard cap: 90 words ─────────────────────────────────
    words = reply.split()
    if len(words) > 90:
        candidate = " ".join(words[:90])
        last_danda = candidate.rfind("।")
        if last_danda > 0:
            reply = candidate[:last_danda + 1]
        else:
            reply = candidate + "।"
        print(
            f"[get_response] \u2702 Word-count truncation ({len(words)} \u2192 \u226490 words).",
            file=sys.stderr,
        )

    # ── Specific Drug / Medicine Name Stripping ────────────────────────────
    # Prohibit specific drug names (e.g. Paracetamol, Ibuprofen, Adrenaline)
    # and replace with generic advisor text.
    DRUG_RE = re.compile(
        r"(?:Paracetamol|Ibuprofen|Adrenaline|Dextrose|पेरासिटामोल|"
        r"পেৰাচিটামল|আইবুপ্ৰফেন|এড্ৰেনালিন|ডেক্সট্ৰ’জ|মেটফৰ্মিন|Metformin)",
        re.IGNORECASE,
    )
    if DRUG_RE.search(reply):
        reply = DRUG_RE.sub("চিকিৎসকৰ পৰামৰ্শ অনুসৰি উপযুক্ত ঔষধ", reply)
        print(
            "[get_response] \u26a0 Specific drug name detected and replaced with generic advice.",
            file=sys.stderr,
        )

    # ── Dosage hallucination guard ─────────────────────────────────────────
    dosage_match = _DOSAGE_PATTERN.search(reply)
    if dosage_match:
        pre = reply[:dosage_match.start()].rstrip(" ,;-")
        sent_end = max(pre.rfind("।"), pre.rfind("."), pre.rfind("?"), pre.rfind("!"))
        reply = (pre[:sent_end + 1] if sent_end > 0 else pre + "।").rstrip()
        reply += " চিকিৎসকৰ পৰামৰ্শ লওক।"
        print(
            "[get_response] \u26a0 Dosage hallucination detected and removed. "
            "Appended doctor-consult redirect.",
            file=sys.stderr,
        )

    # ── Deterministic Bengali → Assamese script correction ─────────────────
    BENGALI_TO_ASSAMESE = {
        "\u09B0": "\u09F0",   # র  → ৰ  (ra)
        "\u09DC": "\u09F0",   # ড় → ৰ
        "\u09DD": "\u09F0",   # ঢ় → ৰ
        "\u09DF": "\u09F1",   # য় → ৱ  (wa)
    }
    corrected = reply
    for bengali_char, assamese_char in BENGALI_TO_ASSAMESE.items():
        corrected = corrected.replace(bengali_char, assamese_char)
    if corrected != reply:
        print(
            "[get_response] \u2139\ufe0f Auto-corrected Bengali script to Assamese.",
            file=sys.stderr,
        )
        reply = corrected

    # Correct word-initial ৱি (which is always spelling drift from Sanskrit/Bengali বি-) to বি
    # e.g., ৱিভাগ -> বিভাগ, ৱিকাস -> বিকাশ, ৱিচ্ছেদ -> বিচ্ছেদ.
    # Uses a negative lookbehind to ensure U+09F1\u09BF (ৱি) starts a word.
    corrected_wi = re.sub(r"(?<![^\s\(\[\-\,\.।])\u09F1\u09BF", "\u09AC\u09BF", reply)
    if corrected_wi != reply:
        print(
            "[get_response] \u2139\ufe0f Corrected word-initial spelling drift (ৱি- \u2192 বি-).",
            file=sys.stderr,
        )
        reply = corrected_wi

    return reply.strip()


# ---------------------------------------------------------------------------
# Translation Engine
# ---------------------------------------------------------------------------

def translate_en_to_as(text: str) -> str:
    """
    Translates English text to Assamese using the free Google Translate API.
    Includes retry logic and safety fallbacks.
    """
    import urllib3
    urllib3.disable_warnings()

    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx",
        "sl": "en",
        "tl": "as",
        "dt": "t",
        "q": text,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    print(f"[translate] Translating English response to Assamese...", file=sys.stderr)

    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=headers, params=params, verify=False, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            translated_chunks = []
            if data and isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                for chunk in data[0]:
                    if isinstance(chunk, list) and len(chunk) > 0 and isinstance(chunk[0], str):
                        translated_chunks.append(chunk[0])
            if translated_chunks:
                translated_text = "".join(translated_chunks)
                return translated_text.strip()
            raise ValueError("Translation format is unexpected.")
        except Exception as e:
            print(f"[translate] Attempt {attempt}/3 failed: {e}", file=sys.stderr)
            if attempt == 3:
                return "অনুগ্ৰহ কৰি আপোনাৰ নিকটৱৰ্তী স্বাস্থ্যকেন্দ্ৰ বা চৰকাৰী কাৰ্যালয়ৰ সৈতে যোগাযোগ কৰক।"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_response(text: str) -> str:
    """
    Returns an Assamese healthcare/governance reply for the given query.

    Pipeline:
      1. Check TEMPLATES — bypass LLM if query matches emergency/demo keywords.
      2. Call NVIDIA NIM LLM to generate an English response (Fallback).
      3. Translate English response to Assamese.
      4. Apply post-processing script corrections & guards.
    """
    # ── Step 1: Template routing (TEMPORARILY BYPASSED FOR TESTING) ───────
    # norm_text = normalize(text)
    # for topic, entry in TEMPLATES.items():
    #     for kw in entry["keywords"]:
    #         if isinstance(kw, tuple):
    #             if all(normalize(sub_kw) in norm_text for sub_kw in kw):
    #                 print(f"[get_response] Template matched: {topic}", file=sys.stderr)
    #                 return entry["answer"]
    #         elif isinstance(kw, str):
    #             if normalize(kw) in norm_text:
    #                 print(f"[get_response] Template matched: {topic}", file=sys.stderr)
    #                 return entry["answer"]
    pass

    # ── Step 1b: DEMO_MODE fallback ───────────────────────────────────────
    if DEMO_MODE:
        print("[get_response] DEMO_MODE active — returning safe fallback (no LLM call)", file=sys.stderr)
        return "এই প্ৰশ্নটোৰ বাবে বিশেষজ্ঞ তথ্য প্ৰয়োজন। অনুগ্ৰহ কৰি আপোনাৰ নিকটৱৰ্তী স্বাস্থ্যকেন্দ্ৰ বা চৰকাৰী কাৰ্যালয়ৰ সৈতে যোগাযোগ কৰক।"

    print("[get_response] No template match — falling back to LLM", file=sys.stderr)

    # ── Step 2: LLM Fallback Call (English output) ────────────────────────
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable is not set.", file=sys.stderr)
        return "Sorry, I couldn't process that. Please try again."

    url     = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    examples_str = (
        "Query: জ্বৰ হ'লে কি কৰিব লাগে? -> Response: If you have fever, you should take plenty of rest, drink enough water, and keep yourself cool. If the fever persists for more than three days, you should consult a doctor.\n"
        "Query: How can I apply for a new ration card? -> Response: To apply for a new ration card, you should visit your nearest food and civil supplies office with your citizenship and address proof documents.\n"
        "Query: Where can I check my voter ID status? -> Response: You can check your voter registration status by visiting the official National Voters' Service Portal website."
    )

    system_prompt = (
        "You are a healthcare and governance assistant for Northeast India, specifically Assam. "
        "Your users will ask questions in Assamese, Hindi, English, or mixed languages. "
        "You must understand their query, but you must respond ONLY in clean, plain English prose. "
        "DO NOT use Assamese script or Hindi script in your response. "
        
        "SCOPE RULE: Only answer questions about healthcare, medical symptoms, government schemes, or governance procedures. "
        "If the question is about anything else, respond ONLY with: "
        "'I can only help with healthcare and governance questions.' "
        "Do not answer off-topic questions even partially. "

        "DRUG & DOSAGE RULE: NEVER state specific drug/medicine names (like Paracetamol, Ibuprofen, Adrenaline, Dextrose) or dosages. "
        "Instead of naming any medicine, give generic advice like 'take rest and appropriate medication as advised by a doctor'. "

        "FORMAT RULE: Write ONLY plain English prose sentences. "
        "ABSOLUTELY NO numbered lists (1. 2. 3.), bullet points, bold (**text**), italic (*text*), "
        "or (i)/(ii)/(iii) markers. Write 2-3 complete sentences, under 60 words.\n\n"

        "Examples of desired output:\n"
        f"{examples_str}\n"
        "Now respond to the user's query in plain English following all rules."
    )

    payload = {
        "model":             MODEL_NAME,
        "messages":          [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
        "temperature":       0.2,
        "max_tokens":        250,      # Increased to 250 to allow complete sentences
        "frequency_penalty": 0.4,
        "presence_penalty":  0.3,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                reply = choice["message"]["content"]
                if reply is not None:
                    # Save raw English response for debugging/tests
                    global last_raw_reply
                    last_raw_reply = reply.strip()
                    # ── Step 3: Translate to Assamese ──────────────────────
                    translated_reply = translate_en_to_as(last_raw_reply)
                    # ── Step 4: Post-processing on translated text ─────────
                    return _postprocess(translated_reply)

        raise ValueError("NVIDIA NIM API response does not contain expected format.")

    except Exception as e:
        print(f"Error calling NVIDIA NIM API: {e}", file=sys.stderr)
        return "Sorry, I couldn't process that. Please try again."
