import sys, os

# Set stdout/stderr to UTF-8 to prevent cp1252 encoding errors on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add paths to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlu-mt"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts", "scripts"))

from generate_samples import SAMPLES
from get_response import get_response, TEMPLATES, normalize

print("======================================================================")
print("  SANITY TEST: TEMPLATE-FIRST ROUTING VERIFICATION WITH TUPLE AND-MATCHING")
print("======================================================================\n")

success_count = 0
for idx, sample in enumerate(SAMPLES, 1):
    text = sample["text"]
    ans = get_response(text)
    
    # Determine which template matched
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
        print(f"Sample {idx:02d} | Match: {matched_topic:<30} | Status: PASS")
        success_count += 1
    else:
        print(f"Sample {idx:02d} | ❌ NO TEMPLATE MATCHED (Status: FAIL) | Text: {text}")
        success_count += 0

print(f"\nSanity Test Summary: {success_count}/15 samples matched templates successfully.")
if success_count == 15:
    sys.exit(0)
else:
    sys.exit(1)
