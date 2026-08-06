import sys
import os
import time
import io

# Reconfigure stdout and stderr to support UTF-8 encoding (especially on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Auto-install dependencies if they are missing
try:
    import datasets
    import soundfile
    import transformers
    import torchaudio
    import onnxruntime
except ImportError:
    print("Required libraries not found. Installing them now...")
    import subprocess
    try:
        # Run pip install using the current python executable to target the virtual environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "soundfile", "transformers", "torchaudio", "onnxruntime"])
        import datasets
        import soundfile
        import transformers
        import torchaudio
        import onnxruntime
        print("Libraries installed successfully.\n")
    except Exception as e:
        print(f"Error: Failed to install dependencies: {e}", file=sys.stderr)
        sys.exit(1)

import torch
import numpy as np
import soundfile as sf
from transformers import AutoModel

def get_hf_token():
    """Retrieve HF_TOKEN from environment variables or .env files."""
    # Check environment variables
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
        
    # Find the project root relative to the script file (asr/scripts/transcribe_test.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Check .env files in common locations relative to project root
    env_paths = [
        os.path.join(project_root, ".env"),
        os.path.join(project_root, "nlu-mt", ".env"),
        os.path.join(project_root, "..", ".env"),
    ]
    for path in env_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("HF_TOKEN="):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                val = parts[1].strip()
                                # Strip enclosing quotes if present
                                if val.startswith(('"', "'")) and val.endswith(val[0]):
                                    val = val[1:-1]
                                return val
            except Exception:
                pass
    return None

def resample_audio(audio_array, orig_sr, target_sr=16000):
    """Resample 1D audio array to target_sr using linear interpolation."""
    if orig_sr == target_sr:
        return audio_array
    duration = len(audio_array) / orig_sr
    num_target_samples = int(duration * target_sr)
    x_orig = np.linspace(0, duration, len(audio_array))
    x_target = np.linspace(0, duration, num_target_samples)
    return np.interp(x_target, x_orig, audio_array)

def main():
    token = get_hf_token()
    
    print("Loading Hugging Face dataset 'google/fleurs' with config 'as_in' (Assamese) train split...")
    try:
        # Load the train split of the google/fleurs dataset with Assamese config.
        # This will reuse the locally cached dataset if available.
        dataset = datasets.load_dataset("google/fleurs", "as_in", split="train", token=token)
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR LOADING DATASET:")
        print(f"{e}")
        print("="*80)
        print("\nPlease make sure you have internet access and that the config name is correct.")
        print("="*80 + "\n")
        sys.exit(1)
        
    # Cast the audio column with decode=False to bypass datasets library's built-in decoding
    print("Casting audio column with decode=False...")
    try:
        dataset = dataset.cast_column("audio", datasets.Audio(decode=False))
    except Exception as e:
        print(f"Warning: Could not cast audio column: {e}")
    
    # Load ai4bharat/indic-conformer-600m-multilingual ASR model
    print("Loading ASR model 'ai4bharat/indic-conformer-600m-multilingual' (this might take a moment)...")
    try:
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual", 
            trust_remote_code=True, 
            token=token
        )
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR LOADING ASR MODEL:")
        print(f"{e}")
        print("="*80)
        print("\nNote: 'ai4bharat/indic-conformer-600m-multilingual' is a gated model.")
        print("Please accept the terms at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual")
        print("and make sure your HF_TOKEN is set in your environment or in a .env file.")
        print("="*80 + "\n")
        sys.exit(1)
        
    print("\nRetrieving and transcribing the first 3 samples from the train split:")
    
    # Select the first 3 samples from the dataset
    first_3_samples = dataset.select(range(3))
        
    # Process and transcribe each sample
    for i, sample in enumerate(first_3_samples):
        print(f"=== Sample {i + 1} ===")
        try:
            # Ground truth text is in the 'transcription' column for google/fleurs
            ground_truth = sample.get("transcription", "")
            
            # Extract raw audio bytes
            audio_info = sample.get("audio", {})
            audio_bytes = audio_info.get("bytes")
            
            if audio_bytes is None:
                print(f"Error: No audio bytes found for sample {i + 1}.", file=sys.stderr)
                continue
                
            # Decode audio bytes using soundfile
            audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
                
            # Ensure audio is float32
            audio_array = audio_array.astype(np.float32)
            
            # Resample to 16kHz if necessary (Indic-Conformer expects 16kHz)
            if sampling_rate != 16000:
                audio_array = resample_audio(audio_array, sampling_rate, 16000)
            
            # Convert numpy array to 2D torch tensor shape (1, samples)
            wav_tensor = torch.tensor(audio_array).unsqueeze(0)
            
            # Run transcription using CTC decoding
            with torch.no_grad():
                predicted_text = model(wav_tensor, "as", "ctc")
                
            # Parse output format safely
            if isinstance(predicted_text, (list, tuple)):
                if len(predicted_text) > 0:
                    predicted_text = predicted_text[0]
                else:
                    predicted_text = ""
            predicted_text = str(predicted_text).strip()
            
            # Print results
            print(f"Ground Truth Text: {ground_truth}")
            print(f"Prediction       : {predicted_text}")
            print("=" * 20 + "\n")
            
        except Exception as e:
            print(f"Error processing sample {i + 1}: {e}\n", file=sys.stderr)

if __name__ == "__main__":
    main()
