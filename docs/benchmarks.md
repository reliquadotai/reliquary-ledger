# Reliquary Ledger — Benchmarks

Append measurements when you run them. Keep this file mostly empty until Week 1–2 of real deployment.

## Benchmark template

For each run, copy this block and fill it in:

```
## Run <YYYY-MM-DD> <short label>
Operator: <initials>
Hardware: <GPU model / driver / CUDA / cuDNN>
Software: torch <version>, transformers <version>, reliquary-inference <git sha>
Model: <ref>
Profile: <env file or tag>

Generation throughput (single prompt):
  input_tokens: <N>
  output_tokens: <N>
  tokens/sec (miner side, end-to-end): <N>

Proof replay (validator side, per completion):
  sketch compute p50: <ms>
  sketch compute p95: <ms>

Window throughput (accepted completions / minute):
  steady state: <N>
  peak observed: <N>

VRAM steady-state: <N> GB
VRAM peak: <N> GB

Notes:
  - <anything surprising>
```

## Reference points (to beat or match)

- Qwen3-4B-Instruct on L4 (24 GB, CUDA 12.x): target ≥ 60 tokens/sec generation, p95 proof replay < 500 ms. Numbers pending first deployment.
- Qwen2.5-7B-Instruct on A10 (24 GB): target ≥ 35 tokens/sec generation, p95 proof replay < 700 ms.
- Qwen2.5-7B-Instruct on A100 (40 GB): target ≥ 90 tokens/sec generation, p95 proof replay < 300 ms.

## Runs

_(append here)_
