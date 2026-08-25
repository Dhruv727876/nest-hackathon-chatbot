"""Audit all 15 SAMPLES for weak word-initial consonants."""
import sys, os
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tts", "scripts"))
from generate_samples import SAMPLES

WEAK = {"স", "শ", "হ"}

print("Auditing all 15 SAMPLES for weak initial consonants (স শ হ):")
print("=" * 65)
at_risk = []
for i, s in enumerate(SAMPLES, 1):
    first_char = s["text"][0]
    risk = first_char in WEAK
    flag = "  ⚠  WEAK START" if risk else ""
    topic = s["topic"]
    text  = s["text"]
    print(f"  [{i:02d}] {first_char}  |  {topic[:45]}{flag}")
    if risk:
        print(f"         text: {text}")
        at_risk.append((i, first_char, topic, text))
print("=" * 65)
print(f"\nAt-risk samples: {[x[0] for x in at_risk]}")
