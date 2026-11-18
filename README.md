# LLM High-Throughput Serving Benchmark (vLLM vs HuggingFace)

A performance engineering benchmark designed to measure the throughput and latency trade-offs of serving Large Language Models (LLMs) at scale. This project pits a naive HuggingFace pipeline against **vLLM**, demonstrating the massive performance gains achieved through **continuous batching** and **PagedAttention**.

## The Problem
Standard LLM serving frameworks suffer from severe memory fragmentation. Because the size of the KV cache grows dynamically as tokens are generated, static memory allocation leads to Out-Of-Memory (OOM) errors or massive underutilization of GPU compute.

## The Solution
This benchmark implements **vLLM**, utilizing:
- **PagedAttention:** Treats KV cache like virtual memory, mapping logical blocks to physical GPU blocks on demand. This reduces memory waste to under 4%.
- **Continuous Batching:** Instead of waiting for the longest sequence in a static batch to finish, the scheduler dynamically injects new requests at the iteration level, maximizing GPU saturation.

## Run the Benchmark

**1. Install Dependencies**
```bash
pip install -r requirements.txt
