import math
import wave
import struct
import os
import sys

# TODO: replace with real import once asr-work is merged
# from asr.speech_to_text import speech_to_text
def speech_to_text(audio_path: str) -> str:
    """
    Placeholder speech-to-text function that takes an audio file path
    and returns a mock transcribed text response.
    """
    try:
        if not audio_path or not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")
        # Mock transcribed text related to healthcare
        return "I have fever, what should I do?"
    except Exception as e:
        print(f"Error in speech_to_text: {e}", file=sys.stderr)
        return None

# TODO: replace with real import once nlu-mt-work is merged
# from nlu_mt.get_response import get_response
def get_response(text: str) -> str:
    """
    Placeholder NLU/chatbot response function that takes a text prompt
    and returns a mock healthcare/governance response.
    """
    try:
        if not text:
            raise ValueError("Input text for get_response is empty or None")
        # Mock healthcare reply
        return "Rest, drink fluids, and see a doctor if fever persists more than 2 days"
    except Exception as e:
        print(f"Error in get_response: {e}", file=sys.stderr)
        raise e

def text_to_speech(text: str, output_path: str) -> str:
    """
    Placeholder text-to-speech function that takes input text and generates
    a mock audio file at the specified output_path. Returns the output_path.
    """
    try:
        if not text:
            raise ValueError("Input text for text_to_speech is empty or None")
            
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        # Generate a simple 1.5-second sine wave beep (440Hz) to create a valid, playable WAV file
        sample_rate = 16000
        duration = 1.5  # seconds
        frequency = 440.0  # Hz (A4)
        
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            
            # Write sine wave samples
            for i in range(int(sample_rate * duration)):
                # Amplitude scaled to fit 16-bit signed integer limits
                value = int(16000.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
                data = struct.pack('<h', value)
                wav_file.writeframesraw(data)
                
        return output_path
    except Exception as e:
        print(f"Error in text_to_speech: {e}", file=sys.stderr)
        raise e
