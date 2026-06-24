# llm-training-systems

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A production-style LLM training harness showcasing **distributed training infrastructure** and **numerical precision engineering** — the systems side of LLM training that makes models scale.

This repo does **not** demonstrate model architecture design. It demonstrates:

| Capability | Implementation |
|---|---|
| Data parallelism | `DistributedDataParallel` (DDP) via torchrun |
| ZeRO-3 style sharding | `FullyShardedDataParallel` (FSDP) with per-block wrapping |
| BF16 / FP16 training | `torch.amp.autocast` + `GradScaler` |
| FP8 training | `torchao.float8.convert_to_float8_training` |
| Gradient accumulation | `no_sync()` to suppress redundant all-reduce/reduce-scatter |
| Throughput profiling | tokens/sec, peak GPU memory per step |

---

## Architecture: DDP vs FSDP Memory Layout

```
DDP (Data Parallelism)                  FSDP (ZeRO-3 Style)
──────────────────────────────────      ──────────────────────────────────────────
GPU 0: ┌──────────────────────────┐     GPU 0: ┌──────────────────────────────┐
       │  Full Params   (100 %)   │            │  Param Shard     (100 % / N) │
       │  Full Grads    (100 %)   │            │  Grad Shard      (100 % / N) │
       │  Full Opt St.  (100 %)   │            │  Opt State Shard (100 % / N) │
       └──────────────────────────┘            └──────────────────────────────┘
GPU 1: identical copy ↑                 GPU 1: different shard ↑
GPU 2: identical copy ↑                 GPU 2: different shard ↑
GPU 3: identical copy ↑                 GPU 3: different shard ↑

Memory per GPU: O(model_size)           Memory per GPU: O(model_size / N)

Communication:                          Communication:
  ▸ All-reduce gradients after bwd        ▸ All-gather params before fwd + bwd
                                          ▸ Reduce-scatter grads after bwd
```

**Key insight:** DDP keeps a full model copy on every GPU — simple but memory-heavy. FSDP shards parameters, gradients, and optimizer state across GPUs, so a 70 B model that needs ~140 GB in BF16 becomes feasible across 8× H100 (17.5 GB/GPU) instead of requiring 8 identical 80 GB cards.

---

## Why FP8 Matters

Modern accelerators (H100, H200, Blackwell) include dedicated FP8 tensor cores. Compared to BF16:

| Metric | BF16 | FP8 |
|---|---|---|
| Bits per weight | 16 | 8 |
| Memory bandwidth | 1× | ~2× |
| FLOPS (H100 SXM) | 989 TFLOPS | 1979 TFLOPS |
| Numerical range | wide | narrow (needs scaling) |

FP8 uses two formats: **E4M3** (higher precision, used for activations) and **E5M2** (wider range, used for gradients). Scaling factors prevent overflow/underflow — this is handled transparently by `torchao`'s `Float8Linear` which replaces `nn.Linear` at the module level.

**When to use:** Any training run on Hopper-class (H100+) hardware where memory bandwidth is the bottleneck — which is almost all transformer training beyond a few hundred million parameters.

```python
# torchao replaces nn.Linear with Float8Linear in-place.
# Weights/activations stored E4M3; gradients stored E5M2.
# Accumulation happens in BF16 for stability.
from torchao.float8 import convert_to_float8_training
convert_to_float8_training(model)
```

---

## Benchmark Results

> **Note:** Fill in with real numbers after running on your hardware. Example below uses A100 80GB × 2.

| Config | Tokens/sec | Peak Mem (GB) | Final Loss |
|---|---|---|---|
| ddp_fp32 | — | — | — |
| ddp_bf16 | — | — | — |
| ddp_fp16 | — | — | — |
| ddp_fp8  | — | — | — |
| fsdp_fp32 | — | — | — |
| fsdp_bf16 | — | — | — |
| fsdp_fp16 | — | — | — |

Run `python benchmark.py` to populate this table and generate `results/benchmark_plot.png`.

---

## Repo Structure

```
llm-training-systems/
├── train.py          # Main training script (DDP/FSDP, all dtypes)
├── benchmark.py      # Orchestrator: runs all combos, renders table + plot
├── launch.sh         # torchrun wrapper with env-var overrides
├── requirements.txt
├── configs/
│   ├── small.yaml    # gpt2, 512 seq_len, 2 GPUs
│   └── medium.yaml   # gpt2-medium, 1024 seq_len, 4 GPUs, FSDP
└── results/          # JSON per run + summary.csv + benchmark_plot.png
```

---

## Setup

```bash
pip install -r requirements.txt
```

FP8 additionally requires CUDA 12.1+ and a Hopper-class GPU. The rest works on any CUDA-capable GPU.

---

## Quickstart

### Single GPU — DDP BF16 (simplest baseline)
```bash
torchrun --nproc_per_node=1 train.py --strategy ddp --dtype bf16 --steps 100
```

### Multi-GPU — DDP BF16
```bash
torchrun --nproc_per_node=4 train.py --strategy ddp --dtype bf16 --steps 200
```

### Multi-GPU — FSDP BF16 (ZeRO-3 style)
```bash
torchrun --nproc_per_node=4 train.py \
    --strategy fsdp \
    --dtype bf16 \
    --model gpt2-medium \
    --batch_size 4 \
    --seq_len 1024 \
    --grad_accum 4
```

### FP8 on H100 (requires torchao + Hopper GPU)
```bash
torchrun --nproc_per_node=1 train.py --strategy ddp --dtype fp8
```

### Load from config file
```bash
NUM_GPUS=2 STRATEGY=fsdp DTYPE=bf16 ./launch.sh --config configs/medium.yaml
```

### Run all benchmarks and generate comparison plots
```bash
python benchmark.py --gpus 2 --model gpt2 --steps 100
```

---

## Key Training Flags

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `ddp` | `ddp` or `fsdp` |
| `--dtype` | `bf16` | `fp32`, `bf16`, `fp16`, `fp8` |
| `--model` | `gpt2` | Any HuggingFace causal LM |
| `--batch_size` | `4` | Per-GPU micro-batch size |
| `--seq_len` | `512` | Sequence length in tokens |
| `--steps` | `100` | Optimizer steps |
| `--grad_accum` | `1` | Gradient accumulation steps |
| `--config` | — | YAML config to load |

Effective global batch size = `batch_size × grad_accum × world_size × seq_len` tokens.

---

## Gradient Accumulation and `no_sync()`

When `--grad_accum > 1`, the inner loop calls `model.no_sync()` on all but the last micro-step:

```
micro-step 0:  forward + backward  ─── no_sync() ──→ gradients stay local
micro-step 1:  forward + backward  ─── no_sync() ──→ gradients stay local
micro-step 2:  forward + backward  ─── sync ──────→ all-reduce (DDP) or
                                                     reduce-scatter (FSDP)
optimizer.step()
```

Without `no_sync()`, DDP triggers an all-reduce after every `.backward()` call, multiplying communication by `grad_accum` — a significant overhead on slow interconnects.

---

## License

MIT
