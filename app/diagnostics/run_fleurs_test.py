import sys
import os
import random
import json
import io
import wave
import difflib

# Set stdout/stderr to UTF-8 to prevent encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Setup paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for p in [os.path.join(PROJECT_ROOT, "asr", "scripts"), os.path.join(PROJECT_ROOT, "nlu-mt"), PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from datasets import Dataset
import datasets as hf_datasets
import soundfile as sf
from asr.scripts.asr_function import speech_to_text

print("======================================================================")
print("  FLEURS ASSAMESE ASR BENCHMARK RUNNER")
print("======================================================================\n")

# 1. Load fleurs-test.arrow
arrow_path = "C:/Users/User/.cache/huggingface/datasets/google___fleurs/as_in/0.0.0/70bb2e84b976b7e960aa89f1c648e09c59f894dd/fleurs-test.arrow"
if not os.path.exists(arrow_path):
    print(f"Error: Cached FLEURS test Arrow file not found at {arrow_path}")
    sys.exit(1)

print("Loading test split from Arrow...")
ds = Dataset.from_file(arrow_path)
ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))
print(f"Loaded {len(ds)} test examples.\n")

# 2. Extract 10 random samples using fixed seed
random.seed(42)
indices = random.sample(range(len(ds)), 10)
print(f"Selected random indices for testing: {indices}\n")

# Make sure app/test_audio/ exists
test_audio_dir = os.path.join(APP_DIR, "test_audio")
os.makedirs(test_audio_dir, exist_ok=True)

ground_truth_map = {}
selected_samples = []

for idx, ds_idx in enumerate(indices, 1):
    sample = ds[ds_idx]
    
    # Save audio to app/test_audio/fleurs_01.wav through fleurs_10.wav
    audio_dict = sample["audio"]
    raw_bytes = audio_dict["bytes"]
    orig_path = audio_dict["path"]
    
    # Read bytes using soundfile
    audio_data, sr = sf.read(io.BytesIO(raw_bytes))
    
    # fleurs_XX.wav
    wav_filename = f"fleurs_{idx:02d}.wav"
    wav_path = os.path.join(test_audio_dir, wav_filename)
    
    # Save to disk as 16kHz mono (already 16kHz mono, but sf.write enforces formatting)
    sf.write(wav_path, audio_data, sr)
    
    transcription = sample.get("transcription", "").strip()
    raw_transcription = sample.get("raw_transcription", "").strip()
    
    # Save metadata
    ground_truth_map[wav_filename] = {
        "transcription": transcription,
        "raw_transcription": raw_transcription,
        "original_index": ds_idx,
        "original_path": orig_path
    }
    
    selected_samples.append((wav_path, wav_filename, transcription))

# Write fleurs_ground_truth.json
gt_json_path = os.path.join(test_audio_dir, "fleurs_ground_truth.json")
with open(gt_json_path, "w", encoding="utf-8") as fh:
    json.dump(ground_truth_map, fh, ensure_ascii=False, indent=2)
print(f"Saved ground truth metadata mapping to {gt_json_path}\n")

# 3. Helper metrics functions
def compute_wer(ref: str, hyp: str) -> float:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    d = [[0 for _ in range(len(hyp_words) + 1)] for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + 1)
    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def word_diff(ref: str, hyp: str) -> str:
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    diff_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for w in ref_words[i1:i2]:
                diff_parts.append(w)
        elif tag == 'replace':
            for w in ref_words[i1:i2]:
                diff_parts.append(f"[-{w}-]")
            for w in hyp_words[j1:j2]:
                diff_parts.append(f"[+{w}+]")
        elif tag == 'delete':
            for w in ref_words[i1:i2]:
                diff_parts.append(f"[-{w}-]")
        elif tag == 'insert':
            for w in hyp_words[j1:j2]:
                diff_parts.append(f"[+{w}+]")
    return " ".join(diff_parts)

# 4. Run ASR step on all 10 files
results = []

for idx, (wav_path, wav_filename, ref_text) in enumerate(selected_samples, 1):
    print(f"Transcribing {wav_filename} ...")
    try:
        asr_output = speech_to_text(wav_path)
    except Exception as e:
        asr_output = f"ERROR: {e}"
        
    wer = compute_wer(ref_text, asr_output)
    diff = word_diff(ref_text, asr_output)
    
    # Accuracy classification
    if "ERROR" in asr_output:
        verdict = "poor match"
    elif wer <= 0.15:
        verdict = "good match"
    elif wer <= 0.50:
        verdict = "partial match"
    else:
        verdict = "poor match"
        
    print(f"  Ground Truth: {ref_text}")
    print(f"  ASR Output:   {asr_output}")
    print(f"  WER:          {wer:.2%}")
    print(f"  Visual Diff:  {diff}")
    print("-" * 75)
    
    results.append({
        "sample": f"{idx:02d}",
        "ref": ref_text,
        "hyp": asr_output,
        "wer": f"{wer:.2%}",
        "diff": diff,
        "verdict": verdict
    })

# 5. Output Summary Table
print("\n" + "="*95)
print("  FLEURS ASSAMESE ASR ACCURACY SUMMARY TABLE")
print("="*95)
print("| Sample | Ground Truth | ASR Output | WER | Verdict |")
print("|--------|--------------|------------|-----|---------|")
for r in results:
    trunc_ref = r['ref'][:25] + "..." if len(r['ref']) > 25 else r['ref']
    trunc_hyp = r['hyp'][:25] + "..." if len(r['hyp']) > 25 else r['hyp']
    print(f"| {r['sample']} | {trunc_ref:<25} | {trunc_hyp:<25} | {r['wer']:<5} | {r['verdict']:<13} |")
print("="*95)
print("\nVisual Diffs:")
for r in results:
    print(f"Sample {r['sample']}: {r['diff']}")
