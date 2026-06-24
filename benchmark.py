"""
Benchmark orchestrator for llm-training-systems.

Runs all strategy × dtype combinations by launching train.py via torchrun,
then loads the resulting JSON files and renders:
  - A rich terminal table (strategy | dtype | tokens/sec | peak_mem_gb | final_loss)
  - results/summary.csv
  - results/benchmark_plot.png  (side-by-side bar charts)

Usage:
    python benchmark.py                     # run all non-fp8 combos, 1 GPU
    python benchmark.py --gpus 2            # 2 GPUs per run
    python benchmark.py --include-fp8       # also bench FP8 (needs H100)
    python benchmark.py --skip-training     # only load existing JSONs and plot
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table


# Default combos (fp8 opt-in because it requires Hopper-class GPU)
_BASE_COMBOS: list[tuple[str, str]] = [
    ("ddp",  "fp32"),
    ("ddp",  "bf16"),
    ("ddp",  "fp16"),
    ("fsdp", "fp32"),
    ("fsdp", "bf16"),
    ("fsdp", "fp16"),
]
_FP8_COMBOS: list[tuple[str, str]] = [
    ("ddp",  "fp8"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM training benchmarks and compare results.")
    p.add_argument("--gpus",           type=int, default=1,       help="GPUs per training run")
    p.add_argument("--model",          default="gpt2",            help="HuggingFace model name")
    p.add_argument("--steps",          type=int, default=100)
    p.add_argument("--batch_size",     type=int, default=4)
    p.add_argument("--seq_len",        type=int, default=512)
    p.add_argument("--grad_accum",     type=int, default=1)
    p.add_argument("--output_dir",     default="results")
    p.add_argument("--include-fp8",    action="store_true",
                   help="Include FP8 combos (requires H100 or newer GPU)")
    p.add_argument("--skip-training",  action="store_true",
                   help="Skip training runs; only visualise existing result JSONs")
    p.add_argument("--force-rerun",    action="store_true",
                   help="Re-run even if a result JSON already exists")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training launcher
# ---------------------------------------------------------------------------

def run_combo(
    strategy: str,
    dtype: str,
    args: argparse.Namespace,
) -> bool:
    """
    Launch one train.py run via torch.distributed.run (torchrun).
    Returns True on success.

    We use `python -m torch.distributed.run` instead of the `torchrun` shell
    wrapper so this works cross-platform without adding torchrun to PATH.
    """
    out_path = Path(args.output_dir) / f"{strategy}_{dtype}.json"
    if out_path.exists() and not args.force_rerun:
        print(f"  [skip] {strategy}_{dtype} — result already exists")
        return True

    print(f"\n{'─'*50}")
    print(f"  Running: strategy={strategy}  dtype={dtype}")
    print(f"{'─'*50}")

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.gpus}",
        "--master_port=29500",
        "train.py",
        "--strategy",    strategy,
        "--dtype",       dtype,
        "--model",       args.model,
        "--steps",       str(args.steps),
        "--batch_size",  str(args.batch_size),
        "--seq_len",     str(args.seq_len),
        "--grad_accum",  str(args.grad_accum),
        "--output_dir",  args.output_dir,
    ]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [FAILED] {strategy}_{dtype} exited with code {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------

def load_results(output_dir: str, combos: list[tuple[str, str]]) -> list[dict]:
    results = []
    for strategy, dtype in combos:
        path = Path(output_dir) / f"{strategy}_{dtype}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
        else:
            print(f"  [missing] No result for {strategy}_{dtype}")
    return results


# ---------------------------------------------------------------------------
# Output: rich table, CSV, plot
# ---------------------------------------------------------------------------

def print_rich_table(results: list[dict]) -> None:
    console = Console()
    table   = Table(title="LLM Training Benchmark Results", show_lines=True)

    table.add_column("Config",        style="bold cyan",  no_wrap=True)
    table.add_column("Tokens/sec",    justify="right",    style="green")
    table.add_column("Peak Mem (GB)", justify="right",    style="yellow")
    table.add_column("Final Loss",    justify="right",    style="magenta")
    table.add_column("World Size",    justify="right")
    table.add_column("Elapsed (s)",   justify="right")

    for r in results:
        table.add_row(
            f"{r['strategy']}_{r['dtype']}",
            f"{r['tokens_per_sec']:,.0f}",
            f"{r['peak_mem_gb']:.3f}",
            f"{r['final_loss']:.4f}",
            str(r["world_size"]),
            f"{r['elapsed_sec']:.1f}",
        )

    console.print()
    console.print(table)


def save_csv(results: list[dict], output_dir: str) -> None:
    if not results:
        return
    out = Path(output_dir) / "summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV  → {out}")


def save_plot(results: list[dict], output_dir: str) -> None:
    if not results:
        return

    configs     = [f"{r['strategy']}\n{r['dtype']}" for r in results]
    tok_per_sec = [r["tokens_per_sec"]               for r in results]
    peak_mem    = [r["peak_mem_gb"]                  for r in results]
    x           = range(len(configs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LLM Training Benchmark", fontsize=14, fontweight="bold")

    bars1 = ax1.bar(x, tok_per_sec, color="steelblue", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, fontsize=9)
    ax1.set_ylabel("Tokens / second")
    ax1.set_title("Throughput")
    ax1.bar_label(bars1, fmt="{:,.0f}", padding=3, fontsize=8)

    bars2 = ax2.bar(x, peak_mem, color="tomato", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=9)
    ax2.set_ylabel("Peak GPU Memory (GB)")
    ax2.set_title("Peak Memory")
    ax2.bar_label(bars2, fmt="{:.2f}", padding=3, fontsize=8)

    plt.tight_layout()
    out = Path(output_dir) / "benchmark_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    combos = _BASE_COMBOS + (_FP8_COMBOS if args.include_fp8 else [])

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        print(f"\nBenchmarking {len(combos)} config(s) on {args.gpus} GPU(s)...")
        for strategy, dtype in combos:
            run_combo(strategy, dtype, args)

    results = load_results(args.output_dir, combos)

    if not results:
        print("\nNo results found. Run without --skip-training first.")
        return

    print_rich_table(results)
    save_csv(results, args.output_dir)
    save_plot(results, args.output_dir)


if __name__ == "__main__":
    main()
