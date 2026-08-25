import sys
import os
import re

# Set stdout/stderr to UTF-8 to prevent cp1252 encoding errors on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add paths to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlu-mt"))

from get_response import get_response, TEMPLATES, normalize

# Define Devanagari range regex, excluding U+0964 (।) and U+0965 (॥)
DEVANAGARI_RE = re.compile(r"[\u0900-\u0963\u0966-\u097F]")

print("======================================================================")
print("  AUDITING ALL TEMPLATE ANSWERS FOR DEVANAGARI LEAKS (EXCLUDING DANDAS)")
print("======================================================================\n")

all_clean = True
results = []

for topic, entry in TEMPLATES.items():
    ans = entry["answer"]
    devanagari_matches = DEVANAGARI_RE.findall(ans)
    
    # Check for Bengali-specific letters not correct in Assamese: র (09b0), ড় (09dc), ঢ় (09dd), য় (09df)
    bengali_chars = []
    for c in ["\u09b0", "\u09dc", "\u09dd", "\u09df"]:
        if c in ans:
            bengali_chars.append(c)
            
    is_fail = bool(devanagari_matches) or bool(bengali_chars)
    status = "FAIL" if is_fail else "PASS"
    
    details = []
    if devanagari_matches:
        details.append(f"Devanagari characters found: {list(set(devanagari_matches))}")
        all_clean = False
    if bengali_chars:
        details.append(f"Bengali characters found: {bengali_chars}")
        all_clean = False
        
    if not details:
        details.append("Clean Assamese script")
        
    results.append({
        "topic": topic,
        "status": status,
        "details": ", ".join(details),
        "text": ans
    })

# Print Results
print(f"{'Topic Key':<35} | {'Status':<6} | {'Details'}")
print("-" * 100)
for r in results:
    print(f"{r['topic']:<35} | {r['status']:<6} | {r['details']}")

print("\n" + "="*70)
print("  SANITY TEST: RUNNING TEMPLATE ROUTING FOR ALL 15 SAMPLES")
print("="*70)

try:
    from generate_samples import SAMPLES
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts", "scripts"))
    from generate_samples import SAMPLES

success_count = 0
for idx, sample in enumerate(SAMPLES, 1):
    text = sample["text"]
    ans = get_response(text)
    
    matched_topic = None
    norm_text = normalize(text)
    for topic, entry in TEMPLATES.items():
        for kw in entry["keywords"]:
            if isinstance(kw, tuple):
                if all(normalize(sub_kw) in norm_text for sub_kw in kw):
                    matched_topic = topic
                    break
            elif isinstance(kw, str):
                if normalize(kw) in norm_text:
                    matched_topic = topic
                    break
        if matched_topic:
            break
            
    if matched_topic:
        print(f"Sample {idx:02d} | Match: {matched_topic:<30} | Routing: PASS")
        success_count += 1
    else:
        print(f"Sample {idx:02d} | Match: None                           | Routing: FAIL")

print(f"\nRouting verification: {success_count}/15 passed.")
if not all_clean:
    sys.exit(1)
else:
    sys.exit(0)
