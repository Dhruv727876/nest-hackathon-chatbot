import sys
import os
import wave
import io

# Set stdout/stderr to UTF-8 to prevent cp1252 encoding errors on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root and modules to sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for p in [os.path.join(PROJECT_ROOT, "asr", "scripts"), os.path.join(PROJECT_ROOT, "nlu-mt"), os.path.join(PROJECT_ROOT, "tts", "scripts"), PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Redirect get_response logs to capture stdout/stderr output
class LogCapture(object):
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = self.buffer
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr

from app.main import run_pipeline
from get_response import TEMPLATES, normalize

print("======================================================================")
print("  PIPELINE EVALUATION RUNNER (ASR → get_response → TTS)")
print("======================================================================\n")

results = []

for idx in range(1, 16):
    input_wav = os.path.join(PROJECT_ROOT, "tts", "outputs", f"sample_{idx:02d}.wav")
    output_wav = os.path.join(APP_DIR, "outputs", f"pipeline_reply_{idx:02d}.wav")
    
    print(f"Running Sample {idx:02d}...")
    print(f"  Input Path:  {input_wav}")
    print(f"  Output Path: {output_wav}")
    
    # Run pipeline with log capturing
    capture = LogCapture()
    with capture:
        try:
            res = run_pipeline(input_wav, output_wav)
            pipeline_error = None
        except Exception as e:
            res = None
            pipeline_error = str(e)
            
    log_content = capture.buffer.getvalue()
    
    # Parse matched template from capture log content
    # e.g., "[get_response] Template matched: topic_01_fever"
    matched_template = "LLM Fallback"
    for line in log_content.splitlines():
        if "Template matched:" in line:
            matched_template = line.split("Template matched:")[-1].strip()
            break
            
    # Measure duration of output wav
    duration = 0.0
    wav_status = "Not Created"
    if os.path.exists(output_wav):
        try:
            with wave.open(output_wav, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / float(rate)
                wav_status = f"Created ({duration:.2f}s)"
        except Exception as e:
            wav_status = f"Read Error: {e}"
            
    # Flag checks
    flagged = False
    details = []
    
    if pipeline_error:
        flagged = True
        details.append(f"Pipeline error: {pipeline_error}")
    else:
        # Check if matched template fell through to LLM fallback
        if matched_template == "LLM Fallback":
            flagged = True
            details.append("ASR fell through to LLM fallback")
        # Check if duration is abnormally short (<1s)
        if duration < 1.0:
            flagged = True
            details.append("Output audio too short (<1s)")
            
    status = "FAIL" if flagged else "PASS"
    details_str = ", ".join(details) if details else "All checks clean"
    
    transcribed = res["transcribed_text"] if res else "N/A"
    reply = res["reply_text"] if res else "N/A"
    
    print(f"  Transcribed: {transcribed}")
    print(f"  Matched Key: {matched_template}")
    print(f"  Reply Text:  {reply}")
    print(f"  Audio Status: {wav_status}")
    print(f"  Verdict:     {status} ({details_str})")
    print("-" * 75)
    
    results.append({
        "sample": f"{idx:02d}",
        "transcribed": transcribed,
        "matched": matched_template,
        "reply": reply,
        "duration": f"{duration:.2f}s" if duration > 0 else "N/A",
        "status": status,
        "details": details_str
    })

# Print final markdown table
print("\n" + "="*75)
print("  FINAL E2E PIPELINE RESULTS TABLE")
print("="*75)
print("| Sample | Matched Key | Duration | Status | Details |")
print("|--------|-------------|----------|--------|---------|")
for r in results:
    print(f"| {r['sample']} | {r['matched']:<25} | {r['duration']:<8} | {r['status']:<6} | {r['details']} |")
print("="*75)
