import os
import shutil
import onnxruntime.quantization as ort_quant

def main():
    # Resolve paths relative to script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_dir = os.path.join(current_dir, "onnx_model")
    quant_dir = os.path.join(current_dir, "onnx_model_quantized")
    
    model_fp32 = os.path.join(onnx_dir, "model.onnx")
    model_quant = os.path.join(quant_dir, "model.onnx")
    
    if not os.path.exists(model_fp32):
        raise FileNotFoundError(
            f"Original ONNX model not found at {model_fp32}. "
            "Please run 'convert_to_onnx.py' first."
        )
        
    print(f"Applying INT8 dynamic quantization to: {model_fp32}")
    os.makedirs(quant_dir, exist_ok=True)
    
    # Run dynamic quantization
    ort_quant.quantize_dynamic(
        model_input=model_fp32,
        model_output=model_quant,
        weight_type=ort_quant.QuantType.QInt8
    )
    
    # Copy tokenizer and metadata configurations to keep directories matching
    print("Copying configuration and tokenizer files...")
    for filename in os.listdir(onnx_dir):
        if filename != "model.onnx":
            src_file = os.path.join(onnx_dir, filename)
            dst_file = os.path.join(quant_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy(src_file, dst_file)
                
    print(f"Quantized INT8 model successfully saved to: {quant_dir}")

if __name__ == "__main__":
    main()
