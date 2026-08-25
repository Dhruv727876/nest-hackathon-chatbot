"""
app/api_test.py
Direct raw API test — no pipeline.
1. Prints model name from env.
2. Lists available models on integrate.api.nvidia.com.
3. Sends a minimal single-message test to the configured model.
Timeout: 120s.
"""
import sys, os, json, requests
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"): sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nlu-mt", ".env")
load_dotenv(_env)

API_KEY    = os.getenv("NVIDIA_API_KEY", "")
MODEL_NAME = os.getenv("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
BASE_URL   = "https://integrate.api.nvidia.com/v1"

if not API_KEY:
    print("ERROR: NVIDIA_API_KEY not set in nlu-mt/.env")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

print("=" * 65)
print(f"  NIM_MODEL_NAME : {MODEL_NAME}")
print(f"  API key prefix : {API_KEY[:8]}...")
print("=" * 65)

# ── Step 1: List available models ─────────────────────────────────────────
print("\n[1] Fetching available models from NIM ...")
try:
    r = requests.get(f"{BASE_URL}/models", headers=HEADERS, timeout=30)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    print(f"    Found {len(models)} model(s):")
    for m in sorted(models):
        marker = "  <<<  CONFIGURED MODEL" if m == MODEL_NAME else ""
        print(f"      {m}{marker}")
    if MODEL_NAME not in models:
        print(f"\n  ⚠  WARNING: '{MODEL_NAME}' is NOT in the available models list!")
    else:
        print(f"\n  ✅  '{MODEL_NAME}' is available on this NIM endpoint.")
except Exception as e:
    print(f"    ERROR listing models: {e}")

# ── Step 2: Raw chat completion test ──────────────────────────────────────
print(f"\n[2] Sending minimal test prompt to '{MODEL_NAME}' (timeout=120s) ...")
payload = {
    "model":       MODEL_NAME,
    "messages":    [{"role": "user", "content": "Hello, reply with one sentence in English."}],
    "temperature": 0.1,
    "max_tokens":  50,
}
try:
    import time
    t0 = time.time()
    r2 = requests.post(f"{BASE_URL}/chat/completions", headers=HEADERS,
                       json=payload, timeout=120)
    elapsed = time.time() - t0
    r2.raise_for_status()
    data    = r2.json()
    content = data["choices"][0]["message"]["content"]
    print(f"    ✅  Response received in {elapsed:.1f}s")
    print(f"    Reply: {content.strip()}")
except requests.exceptions.Timeout:
    print(f"    ❌  Request timed out after 120s.")
    print("        The model exists but is too slow for the current timeout.")
except requests.exceptions.HTTPError as e:
    print(f"    ❌  HTTP error: {e.response.status_code} — {e.response.text[:300]}")
except Exception as e:
    print(f"    ❌  Unexpected error: {e}")

print("\n" + "=" * 65)
