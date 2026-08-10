import os
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def main():
    # Resolve paths relative to script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_model_path = os.path.join(current_dir, "onnx_model")
    
    # Tiny, lightweight BERT model suited for on-device demonstration
    model_id = "prajjwal1/bert-tiny"
    
    print(f"Loading and exporting model '{model_id}' to ONNX...")
    
    # Load and export the PyTorch model to ONNX format
    model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Save the ONNX model files and tokenizer configuration
    os.makedirs(onnx_model_path, exist_ok=True)
    model.save_pretrained(onnx_model_path)
    tokenizer.save_pretrained(onnx_model_path)
    
    print(f"Original ONNX model successfully saved to: {onnx_model_path}")

if __name__ == "__main__":
    main()
