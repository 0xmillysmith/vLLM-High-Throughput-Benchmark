import argparse
import time

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # Lightweight but modern for benchmarking
# We simulate a high-concurrency workload by duplicating prompts
PROMPTS = [
    "Explain the architecture of a transformer model in detail.",
    "Write a Python script to train a simple neural network.",
    "What are the trade-offs between latency and throughput in LLM serving?",
    "Describe the concept of PagedAttention in vLLM.",
    "How does continuous batching improve GPU utilization?"
] * 20  # Total of 100 concurrent requests

def benchmark_huggingface():
    print(f"--- Starting HuggingFace (Standard) Benchmark on {MODEL_NAME} ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    # Standard HF pipeline (processes requests sequentially or static batching)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    start_time = time.time()
    total_tokens = 0
    
    # Naive sequential generation to highlight the lack of continuous batching
    for prompt in PROMPTS:
        output = pipe(prompt, max_new_tokens=100, do_sample=False)
        generated_text = output[0]['generated_text']
        total_tokens += len(tokenizer.encode(generated_text))
        
    elapsed = time.time() - start_time
    print(f"Total Time: {elapsed:.2f} seconds")
    print(f"HF Throughput: {total_tokens / elapsed:.2f} tokens/sec")

def benchmark_vllm():
    print(f"--- Starting vLLM (PagedAttention) Benchmark on {MODEL_NAME} ---")
    from vllm import LLM, SamplingParams
    
    # vLLM automatically utilizes PagedAttention for KV cache management
    llm = LLM(model=MODEL_NAME, max_model_len=2048, dtype="half")
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    
    start_time = time.time()
    
    # vLLM engine handles continuous batching asynchronously under the hood
    outputs = llm.generate(PROMPTS, sampling_params)
    
    total_tokens = sum([len(out.outputs[0].token_ids) for out in outputs])
    elapsed = time.time() - start_time
    
    print(f"Total Time: {elapsed:.2f} seconds")
    print(f"vLLM Throughput: {total_tokens / elapsed:.2f} tokens/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Serving Benchmark")
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], required=True, 
                        help="Choose the backend: 'hf' (HuggingFace) or 'vllm'")
    
    args = parser.parse_args()
    
    if args.backend == "hf":
        benchmark_huggingface()
    elif args.backend == "vllm":
        benchmark_vllm()
