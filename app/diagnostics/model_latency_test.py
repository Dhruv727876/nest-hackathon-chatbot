"""
app/model_latency_test.py
Test candidate models for latency with a short Assamese prompt.
Reports first-response time. Does NOT touch .env.
"""
import sys, os, time, requests

if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlu-mt", ".env")
load_dotenv(_env)

API_KEY  = os.getenv("NVIDIA_API_KEY", "")
BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HEADERS  = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Candidates ordered by preference
CANDIDATES = [
    "nv-mistralai/mistral-nemo-12b-instruct",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "meta/llama-3.1-8b-instruct",   # known-working fallback
]

PROMPT = (
    "তুমি এজন স্বাস্থ্যসেৱা সহায়ক। "
    "সাধাৰণ চৰ্দিৰ দুটা লক্ষণ সংক্ষেপে কোৱা।"   # "Name two symptoms of common cold briefly."
)

TIMEOUT = 45   # seconds per model

print("=" * 65)
print("  NIM Model Latency Test — candidates in priority order")
print(f"  Timeout per model: {TIMEOUT}s")
print("=" * 65)

winner = None

for model in CANDIDATES:
    print(f"\n  Testing: {model} ...")
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": PROMPT}],
        "temperature": 0.2,
        "max_tokens":  80,
    }
    t0 = time.time()
    try:
        r = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
        elapsed = time.time() - t0
        r.raise_for_status()
        reply = r.json()["choices"][0]["message"]["content"].strip()
        print(f"  ✅  Responded in {elapsed:.1f}s")
        print(f"  Reply preview: {reply[:120]}")
        if winner is None:
            winner = model
            print(f"  >>> FIRST PASSING MODEL — will use this one <<<")
            break   # stop at first success
    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        print(f"  ❌  Timed out after {elapsed:.0f}s — skipping")
    except requests.exceptions.HTTPError as e:
        print(f"  ❌  HTTP {e.response.status_code}: {e.response.text[:200]}")
    except Exception as e:
        print(f"  ❌  Error: {e}")

print("\n" + "=" * 65)
if winner:
    print(f"  RECOMMENDATION: Set NIM_MODEL_NAME={winner}")
else:
    print("  All candidates timed out. Revert to meta/llama-3.1-8b-instruct.")
print("=" * 65)
