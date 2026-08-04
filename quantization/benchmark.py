import os
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def main():
    # Resolve paths relative to script location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_fp32_path = os.path.join(current_dir, "onnx_model", "model.onnx")
    model_quant_path = os.path.join(current_dir, "onnx_model_quantized", "model.onnx")
    tokenizer_path = os.path.join(current_dir, "onnx_model")
    
    if not os.path.exists(model_fp32_path) or not os.path.exists(model_quant_path):
        raise FileNotFoundError(
            "ONNX models are missing. Please execute both "
            "'convert_to_onnx.py' and 'quantize.py' before running benchmarks."
        )
        
    print("Initializing benchmark...")
    
    # 1. Measure model file sizes (MB)
    size_fp32 = os.path.getsize(model_fp32_path) / (1024 * 1024)
    size_quant = os.path.getsize(model_quant_path) / (1024 * 1024)
    
    # 2. Setup tokenizer and dummy inference input
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dummy_text = "I have fever, what should I do?"
    inputs = tokenizer(dummy_text, return_tensors="np")
    
    # Convert tokenized outputs to standard ONNX types (np.int64)
    input_feed = {k: v.astype(np.int64) for k, v in inputs.items()}
    
    # 3. Setup CPU Execution Provider Sessions
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session_fp32 = ort.InferenceSession(model_fp32_path, sess_options, providers=["CPUExecutionProvider"])
    session_quant = ort.InferenceSession(model_quant_path, sess_options, providers=["CPUExecutionProvider"])
    
    # Warm-up runs to initialize session graph/cache
    print("Warming up inference sessions...")
    for _ in range(5):
        _ = session_fp32.run(None, input_feed)
        _ = session_quant.run(None, input_feed)
        
    # Benchmark FP32 Model Latency
    print("Benchmarking FP32 model (20 runs)...")
    runs = 20
    latencies_fp32 = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session_fp32.run(None, input_feed)
        latencies_fp32.append((time.perf_counter() - start) * 1000) # milliseconds
        
    avg_latency_fp32 = np.mean(latencies_fp32)
    
    # Benchmark INT8 Model Latency
    print("Benchmarking INT8 Quantized model (20 runs)...")
    latencies_quant = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session_quant.run(None, input_feed)
        latencies_quant.append((time.perf_counter() - start) * 1000) # milliseconds
        
    avg_latency_quant = np.mean(latencies_quant)
    
    # 4. Print Comparison Report Table
    print("\n" + "="*70)
    print("           ONNX QUANTIZATION BENCHMARK REPORT (CPU PROXY)")
    print("="*70)
    print(f"{'Performance Metric':<25} | {'FP32 Model':<12} | {'INT8 Quantized':<14} | {'Change':<10}")
    print("-"*70)
    
    size_change = ((size_quant - size_fp32) / size_fp32) * 100
    print(f"{'Model Size (MB)':<25} | {size_fp32:<12.3f} | {size_quant:<14.3f} | {size_change:+.2f}%")
    
    latency_change = ((avg_latency_quant - avg_latency_fp32) / avg_latency_fp32) * 100
    print(f"{'Avg Latency (ms)':<25} | {avg_latency_fp32:<12.3f} | {avg_latency_quant:<14.3f} | {latency_change:+.2f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
