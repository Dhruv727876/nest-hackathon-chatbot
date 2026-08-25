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

# Redirect stderr to capture template match prints
class LogCapture(object):
    def __init__(self):
        self.buffer = io.StringIO()
        self.original_stderr = sys.stderr
        
    def __enter__(self):
        sys.stderr = self.buffer
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr

import get_response
from app.main import run_pipeline

print("======================================================================")
print("  E2E PIPELINE RUNNER: TIGHTENED NLU QUALITY CHECK VERIFICATION")
print("======================================================================\n")

results = []

for idx in range(1, 11):
    input_wav = os.path.join(APP_DIR, "test_audio", f"live_offtemplate_{idx:02d}.wav")
    output_wav = os.path.join(APP_DIR, "test_audio", f"live_offtemplate_reply_{idx:02d}.wav")
    
    print(f"Processing File live_offtemplate_{idx:02d}.wav...")
    
    # Reset debug globals
    get_response.last_raw_reply = "N/A"
    get_response.last_qc_triggered = False
    
    capture = LogCapture()
    with capture:
        try:
            res = run_pipeline(input_wav, output_wav)
            pipeline_error = None
        except Exception as e:
            res = None
            pipeline_error = str(e)
            
    log_content = capture.buffer.getvalue()
    
    # Read the trace variables
    raw_reply = get_response.last_raw_reply
    qc_triggered = get_response.last_qc_triggered
    
    # Parse reason from stderr logs
    qc_reason = "N/A"
    if qc_triggered:
        for line in log_content.splitlines():
            if "Quality check fail:" in line:
                qc_reason = line.split("Quality check fail:")[-1].strip()
                break
                
    # Measure duration
    duration = 0.0
    if os.path.exists(output_wav):
        try:
            with wave.open(output_wav, 'rb') as w:
                frames = w.getnframes()
                rate = w.getframerate()
                duration = frames / float(rate)
        except:
            pass
            
    transcribed = res["transcribed_text"] if res else "N/A"
    final_reply = res["reply_text"] if res else "N/A"
    
    print(f"  Transcribed: {transcribed}")
    print(f"  Raw LLM:     {raw_reply}")
    print(f"  QC Triggered: {qc_triggered} ({qc_reason})")
    print(f"  Final Reply: {final_reply}")
    print(f"  Audio Dur:   {duration:.2f}s")
    if pipeline_error:
        print(f"  Error:       {pipeline_error}")
    print("-" * 75)
    
    results.append({
        "sample": f"{idx:02d}",
        "transcribed": transcribed,
        "raw_reply": raw_reply,
        "qc_triggered": str(qc_triggered),
        "qc_reason": qc_reason,
        "final_reply": final_reply,
        "duration": f"{duration:.2f}s" if duration > 0 else "N/A"
    })

# Print final results table
print("\n" + "="*95)
print("  QUALITY CHECK AUDIT TABLE (DEMO_MODE=True + Tightened QC Filter)")
print("="*95)
print("| File | Transcription | QC Triggered | Trigger Reason | Final Verdict | Dur |")
print("|------|---------------|--------------|----------------|---------------|-----|")
for r in results:
    trunc_trans = r['transcribed'][:25] + "..." if len(r['transcribed']) > 25 else r['transcribed']
    verdict = "Safe Fallback" if r['qc_triggered'] == "True" else "Passed QC (Generative)"
    print(f"| {r['sample']} | {trunc_trans:<25} | {r['qc_triggered']:<12} | {r['qc_reason']:<30} | {verdict:<22} | {r['duration']:<5} |")
print("="*95)
