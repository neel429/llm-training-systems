"""
LLM training harness demonstrating:
  - DistributedDataParallel (DDP) and FullyShardedDataParallel (FSDP / ZeRO-3)
  - Mixed precision: BF16, FP16 via torch.amp autocast, FP8 via torchao
  - Gradient accumulation with no_sync() to avoid redundant all-reduce/reduce-scatter
  - Per-step throughput and peak-GPU-memory profiling

Launch with torchrun:
    torchrun --nproc_per_node=NUM_GPUS train.py --strategy ddp --dtype bf16
"""

import argparse
import functools
import importlib
import json
import os
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from ring_all_reduce import ring_all_reduce
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# Maps HuggingFace model_type → (module_path, block_class_name) for FSDP wrapping.
# FSDP shards at module boundaries; wrapping at the transformer block level gives
# per-layer sharding that matches ZeRO-3 semantics.
_BLOCK_CLASS_MAP: dict[str, tuple[str, str]] = {
    "gpt2":    ("transformers.models.gpt2.modeling_gpt2",         "GPT2Block"),
    "llama":   ("transformers.models.llama.modeling_llama",        "LlamaDecoderLayer"),
    "mistral": ("transformers.models.mistral.modeling_mistral",    "MistralDecoderLayer"),
    "opt":     ("transformers.models.opt.modeling_opt",            "OPTDecoderLayer"),
    "bloom":   ("transformers.models.bloom.modeling_bloom",        "BloomBlock"),
    "falcon":  ("transformers.models.falcon.modeling_falcon",      "FalconDecoderLayer"),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a HuggingFace LLM with configurable parallelism and precision."
    )
    p.add_argument("--config",      type=str,            help="Path to YAML config (overrides defaults)")
    p.add_argument("--strategy",    choices=["ddp", "fsdp"], default="ddp")
    p.add_argument("--dtype",       choices=["fp32", "bf16", "fp16", "fp8"], default="bf16")
    p.add_argument("--model",       default="gpt2",      help="HuggingFace model name or local path")
    p.add_argument("--seq_len",     type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=4,  help="Per-GPU micro-batch size")
    p.add_argument("--steps",       type=int, default=100)
    p.add_argument("--grad_accum",  type=int, default=1,
                   help="Gradient accumulation steps. Effective BS = batch_size × grad_accum × world_size")
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,  default=0,
                   help="Linear LR warmup steps from 0 → lr")
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--seed",        type=int,  default=42,
                   help="Base RNG seed; each rank gets seed+rank for data diversity")
    p.add_argument("--use_custom_allreduce", action="store_true",
                   help="Replace DDP's built-in all-reduce with the custom ring all-reduce (DDP only)")
    return p.parse_args()


def merge_yaml_config(args: argparse.Namespace) -> argparse.Namespace:
    """Load YAML config and override CLI defaults. CLI flags that were explicitly
    set should take priority, but argparse doesn't expose that directly, so YAML
    wins over built-in defaults — use flags to override YAML on the command line."""
    if args.config is None:
        return args
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    """Initialise NCCL process group. Returns (rank, local_rank, world_size)."""
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def get_block_class(model: nn.Module, model_name: str) -> type:
    """
    Identify the transformer block class for FSDP's ModuleWrapPolicy.

    I first look up the known map; if the model type isn't listed I fall back
    to finding the most frequently repeated non-primitive module class, which is
    almost always the transformer block (repeated once per layer).
    """
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, "model_type", "")

    if model_type in _BLOCK_CLASS_MAP:
        mod_path, cls_name = _BLOCK_CLASS_MAP[model_type]
        return getattr(importlib.import_module(mod_path), cls_name)

    # Generic fallback
    _primitives = {
        "Linear", "LayerNorm", "RMSNorm", "Embedding", "Dropout",
        "GELU", "ReLU", "SiLU", "ModuleList", "Sequential",
        type(model).__name__,
    }
    counts = Counter(
        type(m).__name__
        for m in model.modules()
        if type(m).__name__ not in _primitives
    )
    for cls_name, _ in counts.most_common():
        for m in model.modules():
            if type(m).__name__ == cls_name:
                return type(m)

    raise RuntimeError(
        f"Cannot identify transformer block class for '{model_name}'. "
        "Add its model_type to _BLOCK_CLASS_MAP in train.py."
    )


def build_model(args: argparse.Namespace) -> nn.Module:
    """
    Load the base model in FP32, then optionally apply FP8 conversion.
    """
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    if args.dtype == "fp8":
        try:
            from torchao.float8 import convert_to_float8_training
        except ImportError as e:
            print(f"FP8 import failed: {e}")
            raise
        model = model.cuda()
        
        convert_to_float8_training(model)
    else:
        model = model.cuda()

    return model


def wrap_model(model: nn.Module, args: argparse.Namespace, local_rank: int) -> nn.Module:
    """Apply DDP or FSDP parallelism to the model."""
    if args.strategy == "ddp":
        return DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- FSDP (ZeRO-3 style) ----
    # MixedPrecision tells FSDP which dtype to store params, reduce gradients,
    # and store buffers in. FP8 and FP32 don't need this (fp8 is per-layer).
    mp_cfg: MixedPrecision | None = None
    if args.dtype == "bf16":
        mp_cfg = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif args.dtype == "fp16":
        # Reduce in FP32 for numerical stability; params/buffers live in FP16.
        mp_cfg = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float16,
        )

    block_cls = get_block_class(model, args.model)
    return FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({block_cls}),
        mixed_precision=mp_cfg,
        device_id=local_rank,
    )


def make_autocast_ctx(args: argparse.Namespace):
    """
    Return the autocast context for the forward pass.
    """
    if args.strategy == "ddp":
        if args.dtype == "bf16":
            return torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if args.dtype == "fp16":
            return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


# ---------------------------------------------------------------------------
# Data streaming
# ---------------------------------------------------------------------------

class TokenStream:
    """
    Streams wikitext-103-v1 from HuggingFace datasets, tokenises on the fly,
    and yields fixed-length token chunks for causal LM training.

    Yes, Streaming adds network latency per batch, but at my scale the GPU 
    compute time dominates so it doesn't matter.
    """

    def __init__(self, model_name: str, seq_len: int, rank: int, world_size: int):
        self.model_name = model_name
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        self._tok       = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._gen = self._make_gen()

    def _make_gen(self):
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)
        buf: list[int] = []
        for idx, ex in enumerate(ds):
            if idx % self.world_size != self.rank:
                continue
            text = ex["text"].strip()
            if not text:
                continue
            buf.extend(self._tok.encode(text))
            while len(buf) >= self.seq_len:
                yield buf[: self.seq_len]
                buf = buf[self.seq_len :]

    def next_batch(self, batch_size: int) -> torch.Tensor:
        """Return a (batch_size, seq_len) int64 CPU tensor."""
        rows = []
        for _ in range(batch_size):
            try:
                rows.append(next(self._gen))
            except StopIteration:
                self._gen = self._make_gen()
                rows.append(next(self._gen))
        return torch.tensor(rows, dtype=torch.long)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print(
            f"\n{'='*60}\n"
            f"  strategy={args.strategy}  dtype={args.dtype}  "
            f"model={args.model}  gpus={world_size}\n"
            f"  batch={args.batch_size}  seq_len={args.seq_len}  "
            f"grad_accum={args.grad_accum}  steps={args.steps}\n"
            f"{'='*60}"
        )

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    stream = TokenStream(args.model, args.seq_len, rank, world_size)
    model  = build_model(args)
    model  = wrap_model(model, args, local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = (
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=args.warmup_steps
        )
        if args.warmup_steps > 0 else None
    )

    # GradScaler is only needed for FP16 + DDP. BF16 doesn't overflow in the
    # same way FP16 does, and FSDP FP16 uses its own ShardedGradScaler internally.
    use_scaler = (args.dtype == "fp16" and args.strategy == "ddp")
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    autocast_ctx = make_autocast_ctx(args)

    # Custom ring all-reduce is only meaningful for DDP with >1 GPU.
    use_custom_reduce = (
        args.use_custom_allreduce
        and args.strategy == "ddp"
        and world_size > 1
    )

    torch.cuda.reset_peak_memory_stats(local_rank)
    total_tokens = 0
    total_loss   = 0.0
    t_start      = time.perf_counter()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for accum_idx in range(args.grad_accum):
            input_ids = stream.next_batch(args.batch_size).cuda(non_blocking=True)

            # Suppress gradient synchronisation on all but the last micro-step.
            # For DDP this skips the all-reduce; for FSDP it skips reduce-scatter.
            # This reduces inter-GPU traffic by (grad_accum-1)/grad_accum.
            # When using the custom ring all-reduce, suppress DDP's built-in
            # all-reduce on every step — we will drive it manually below.
            is_last = accum_idx == args.grad_accum - 1
            if use_custom_reduce:
                sync_ctx = model.no_sync()
            else:
                sync_ctx = nullcontext() if is_last else model.no_sync()

            with sync_ctx, autocast_ctx:
                out  = model(input_ids=input_ids, labels=input_ids)
                loss = out.loss / args.grad_accum   # normalise before accumulating

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_loss += loss.item()

        # Manual gradient synchronisation via custom ring all-reduce.
        # Mirrors DDP semantics: SUM across ranks then divide by world_size.
        if use_custom_reduce:
            if scaler is not None:
                scaler.unscale_(optimizer)  # de-scale before touching raw grads
            for param in model.module.parameters():
                if param.grad is not None:
                    ring_all_reduce(param.grad, rank, world_size)
                    param.grad.div_(world_size)

        # Gradient clipping. FSDP.clip_grad_norm_ computes the global norm
        # across shards correctly; torch.nn.utils version only sees local shard.
        already_unscaled = use_custom_reduce and scaler is not None
        if scaler is not None:
            if not already_unscaled:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        elif isinstance(model, FSDP):
            grad_norm = model.clip_grad_norm_(1.0)
            optimizer.step()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        # Tokens processed across ALL ranks this step
        tokens_this_step = args.batch_size * args.seq_len * args.grad_accum * world_size
        total_tokens += tokens_this_step
        total_loss   += step_loss

        if rank == 0 and step % 10 == 0:
            elapsed     = time.perf_counter() - t_start
            tok_per_sec = total_tokens / elapsed
            peak_gb     = torch.cuda.max_memory_allocated(local_rank) / 1e9
            print(
                f"  step {step:4d}/{args.steps}"
                f"  loss={step_loss * args.grad_accum:.4f}"
                f"  grad_norm={grad_norm:.3f}"
                f"  tok/s={tok_per_sec:,.0f}"
                f"  peak_mem={peak_gb:.2f} GB"
            )

    # ---- Persist results ----
    elapsed     = time.perf_counter() - t_start
    tok_per_sec = total_tokens / elapsed
    peak_gb     = torch.cuda.max_memory_allocated(local_rank) / 1e9
    avg_loss    = total_loss / args.steps * args.grad_accum

    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        results = {
            "strategy":      args.strategy,
            "dtype":         args.dtype,
            "model":         args.model,
            "world_size":    world_size,
            "batch_size":    args.batch_size,
            "seq_len":       args.seq_len,
            "steps":         args.steps,
            "grad_accum":    args.grad_accum,
            "tokens_per_sec": round(tok_per_sec, 1),
            "peak_mem_gb":   round(peak_gb, 3),
            "final_loss":    round(avg_loss, 4),
            "elapsed_sec":   round(elapsed, 1),
        }
        out_path = Path(args.output_dir) / f"{args.strategy}_{args.dtype}.json"
        out_path.write_text(json.dumps(results, indent=2))

        print(
            f"\nDone. tok/s={tok_per_sec:,.0f}  "
            f"peak_mem={peak_gb:.2f} GB  avg_loss={avg_loss:.4f}"
        )
        print(f"Results → {out_path}")

    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    args = merge_yaml_config(args)
    train(args)


if __name__ == "__main__":
    main()
