# llm-training-systems

A focused benchmark of distributed LLM training strategies using PyTorch — built to demonstrate production-level understanding of training infrastructure, parallelism, and numerical precision.

![Python](https://img.shields.io/badge/python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange) ![License](https://img.shields.io/badge/license-MIT-green)

---

## What this demonstrates

- **DDP** (DistributedDataParallel) — data-parallel training with NCCL AllReduce gradient sync across GPUs
- **FSDP** (FullyShardedDataParallel) — ZeRO-3 style sharding of model weights, gradients, and optimizer states across GPUs
- **Ring AllReduce from scratch** — custom two-phase (reduce-scatter + allgather) implementation over raw `dist.send/recv`, wired into the DDP training loop as a drop-in replacement for NCCL's built-in `dist.all_reduce`
- **BF16 mixed precision** — automatic mixed precision via `torch.amp` for memory-efficient training
- **Throughput & memory benchmarking** — tokens/sec, peak GPU memory, and loss tracked per run and saved to JSON

---

## Benchmark Results

Ran on **2× NVIDIA A100 SXM4 80GB** (NVLink interconnect) on Vast.ai. Throughput compared at step 50 across all runs for apples-to-apples comparison.  
Model: `gpt2` (~117M params) | Sequence length: 512 | Batch size: 4 | Steps: 50

| Strategy | Dtype | GPUs | Tokens/sec | Peak Mem/GPU | Final Loss |
|----------|-------|------|-----------|--------------|------------|
| DDP | BF16 | 1 | 19,092 | 5.39 GB | 3.56 |
| DDP | BF16 | 2 | 32,474 | 5.39 GB | 3.34 |
| FSDP | BF16 | 2 | 30,837 | 3.45 GB | 3.34 |

### Key takeaways

**DDP 1→2 GPU: 1.70× speedup** — realistic scaling enabled by NVLink's 600 GB/s GPU-to-GPU bandwidth, which keeps NCCL AllReduce overhead minimal relative to compute.

**FSDP vs DDP: 36% memory reduction at 7% throughput cost** — FSDP shards model parameters, gradients, and optimizer states across GPUs. Each GPU holds only `1/N` of the model at rest, reconstructing full layers on-demand via AllGather during forward/backward. At GPT-2 scale the memory saving is modest; at 70B+ scale it's what makes training possible on finite hardware.

---

## Architecture

### DDP — Data Parallel (each GPU holds full model)
```
┌─────────────────────────────────────────────┐
│                  torchrun                    │
│         (spawns 1 process per GPU)           │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐           ┌────▼───┐
│ GPU 0  │           │ GPU 1  │
│ Full   │           │ Full   │
│ Model  │           │ Model  │
│ Batch A│           │ Batch B│
└───┬────┘           └────┬───┘
    │    NCCL AllReduce   │
    │  (average grads)    │
    └─────────┬───────────┘
              │
     Identical weight update
```

### Ring AllReduce — custom reduce-scatter + allgather over raw send/recv
```
Phase 1: Reduce-Scatter (world_size-1 steps)
  Each rank sends its chunk to its right neighbour and accumulates the chunk
  arriving from its left neighbour. After N-1 steps every rank holds one
  fully-summed chunk.

  rank 0 ──[chunk 0]──► rank 1 ──[chunk 1]──► rank 2 ──[chunk 2]──► rank 0
         ◄──[chunk 2]──        ◄──[chunk 0]──        ◄──[chunk 1]──

Phase 2: AllGather (world_size-1 steps)
  Each rank relays its finished chunk around the ring until every rank has
  all chunks. Chunks are overwritten in-place (no accumulation needed).

  rank 0 ──[sum 0]──► rank 1 ──[sum 1]──► rank 2 ──[sum 2]──► rank 0

Result: identical to dist.all_reduce(..., op=SUM) ÷ world_size
```

### FSDP — Fully Sharded (model split across GPUs)
```
┌─────────────────────────────────────────────┐
│                  torchrun                    │
└──────────────┬──────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐           ┌────▼───┐
│ GPU 0  │           │ GPU 1  │
│ Shard 0│◄─AllGather─►Shard 1│  ← forward: reconstruct layer
│ Grad 0 │◄─ReduceScatter──►  │  ← backward: scatter grad shards
│ Opt  0 │           │ Opt  1 │  ← optimizer: each owns its shard
└────────┘           └────────┘
  2.5 GB               2.5 GB    (vs 5 GB each with DDP)
```

---

## Why FSDP over DDP for large models

DDP keeps a full model replica on every GPU. For a 70B parameter model in BF16, that's ~140 GB per GPU — impossible on an 80 GB A100. FSDP shards everything: with 8 GPUs you're down to ~17.5 GB per GPU for weights alone, before accounting for optimizer state sharding (which cuts another ~2–3× on top).

The tradeoff is communication: FSDP does AllGather before every layer's forward and backward pass, then ReduceScatter to redistribute gradients. This is why you see a ~7% throughput penalty vs DDP at small scale — the extra comms add up. At large scale the math flips: DDP becomes impossible (OOM), and FSDP is the only viable option.

---

## Quickstart

### Requirements
```bash
pip install -r requirements.txt
```

### Single GPU (sanity check)
```bash
torchrun --nproc_per_node=1 train.py --strategy ddp --dtype bf16 --steps 50
```

### Multi-GPU DDP
```bash
torchrun --nproc_per_node=2 train.py --strategy ddp --dtype bf16 --steps 100
```

### Multi-GPU DDP with custom ring all-reduce
```bash
torchrun --nproc_per_node=2 train.py --strategy ddp --dtype bf16 --steps 100 --use_custom_allreduce
```

### Multi-GPU FSDP
```bash
torchrun --nproc_per_node=2 train.py --strategy fsdp --dtype bf16 --steps 100
```

### Run full benchmark
```bash
python benchmark.py
# outputs results/summary.csv and results/benchmark_plot.png
```

### All flags
```
--strategy             ddp | fsdp
--dtype                bf16 | fp16 | fp32 | fp8
--model                any HuggingFace causal LM (default: gpt2)
--steps                number of training steps (default: 100)
--batch_size           per-GPU batch size (default: 4)
--seq_len              sequence length (default: 512)
--grad_accum           gradient accumulation steps (default: 1)
--use_custom_allreduce replace NCCL all_reduce with custom ring all-reduce (DDP only)
```

---

## Stack

- **PyTorch 2.12** — DDP, FSDP, AMP
- **NCCL** — GPU-to-GPU collective communications (AllReduce, AllGather, ReduceScatter)
- **HuggingFace Transformers** — model loading
- **HuggingFace Datasets** — streaming wikitext-103
- **torchao** — quantization utilities
- **torchrun** — multi-process launcher

---

## Notes on FP8

FP8 training via `torchao` requires compute capability ≥ 8.9 (RTX 4090) or ≥ 9.0 (H100). The A100 is compute capability 8.0 and does not support `torch._scaled_mm` natively. FP8 emulation mode is possible but doesn't reflect real hardware performance — so BF16 benchmarks are reported here as the honest baseline.

## A note on process

This project was built with the assistance of AI at various stages — code generation (some parts), debugging, and architecture decisions. However, every piece of generated code was read, understood, and validated before being used. Nothing was blindly copy-pasted.

The benchmarks were run on a real rented GPU (2× A100 SXM4 80GB on Vast.ai), not simulated or fabricated. The numbers were then cross-checked for correctness — for example, an initially reported 2.48× scaling figure was identified as a measurement artifact (different step counts between runs) and corrected to the honest 1.70× step-matched comparison.

The goal was to actually understand and practice using a distributed training infrastructure.

## Ring AllReduce — implementation notes

`ring_all_reduce.py` implements the classic two-phase algorithm from scratch using only `dist.send` and `dist.recv`:

- **Reduce-scatter phase** (`world_size - 1` steps): each rank sends its current chunk to its right neighbour and accumulates the incoming chunk from its left. After the phase, each rank holds exactly one fully-summed chunk.
- **AllGather phase** (`world_size - 1` steps): the finished chunks are relayed around the ring. Each rank overwrites (not accumulates) the incoming chunk — it's already a complete sum.
- **Padding**: tensors whose element count isn't a multiple of `world_size` are zero-padded before splitting and trimmed after reconstruction. Zeros are identity elements for SUM, so they don't affect results.
- **Deadlock avoidance**: even ranks send-then-recv; odd ranks recv-then-send. This breaks the simultaneous-send deadlock that would occur if every rank blocked on `send` before calling `recv`.

The function is wired into `train.py` via `--use_custom_allreduce`. When the flag is set, DDP's built-in gradient sync is suppressed with `no_sync()` on every accumulation step, and `ring_all_reduce` is called manually on each parameter's `.grad` tensor after the backward pass, followed by a `div_(world_size)` to produce mean semantics — identical to what NCCL does internally.
