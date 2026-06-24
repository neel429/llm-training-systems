#!/usr/bin/env bash
# Convenience wrapper around torchrun for launching train.py.
#
# Environment overrides (all optional):
#   NUM_GPUS  — number of GPUs per node (default: 1)
#   STRATEGY  — ddp | fsdp              (default: ddp)
#   DTYPE     — fp32 | bf16 | fp16 | fp8 (default: bf16)
#
# Examples:
#   ./launch.sh
#   NUM_GPUS=4 STRATEGY=fsdp DTYPE=bf16 ./launch.sh --steps 500
#   ./launch.sh --config configs/medium.yaml

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-1}"
STRATEGY="${STRATEGY:-ddp}"
DTYPE="${DTYPE:-bf16}"

echo "Launching: strategy=${STRATEGY} dtype=${DTYPE} gpus=${NUM_GPUS}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29500 \
    train.py \
    --strategy "${STRATEGY}" \
    --dtype    "${DTYPE}" \
    "$@"
