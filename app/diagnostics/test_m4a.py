import sys
import os

# Set stdout/stderr to UTF-8 to prevent cp1252 encoding errors on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "asr", "scripts"))

from asr_function import speech_to_text

print("Testing direct MP4 loading on ASR...")
try:
    res = speech_to_text("app/test_audio/live_offtemplate_01.mp4")
    print("Success! Transcribed:", res)
except Exception as e:
    print("Failed as expected:", e)
