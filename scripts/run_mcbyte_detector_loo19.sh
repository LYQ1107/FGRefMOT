#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIXSORT_ROOT="${PROJECT_ROOT}/third_party/MixSort"
PYTHON_BIN="${PYTHON_BIN:-/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019/bin/python}"
DATA_ROOT="${RMOT_MCBYTE_DETECTOR_DATA:-/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_leakage_safe_detector_loo19_20261020}"
OUTPUT_ROOT="${RMOT_MCBYTE_DETECTOR_OUTPUT:-/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/mcbyte_yolox_x_loo19_train_20261020}"
LOCAL_INIT_CKPT="/data3/liuyeqiang/.cache/fgrefmot/yolox_x_coco_init.pth"
if [[ -s "${LOCAL_INIT_CKPT}" ]]; then
  DEFAULT_INIT_CKPT="${LOCAL_INIT_CKPT}"
else
  DEFAULT_INIT_CKPT="${DATA_ROOT}/yolox_x_coco_init.pth"
fi
INIT_CKPT="${RMOT_MCBYTE_DETECTOR_INIT:-${DEFAULT_INIT_CKPT}}"
GPU_COUNT="${RMOT_MCBYTE_DETECTOR_GPU_COUNT:-8}"
BATCH_SIZE="${RMOT_MCBYTE_DETECTOR_BATCH_SIZE:-48}"

export RMOT_MCBYTE_DETECTOR_DATA="${DATA_ROOT}"
export RMOT_MCBYTE_DETECTOR_OUTPUT="${OUTPUT_ROOT}"
export YOLOX_DATADIR="${DATA_ROOT}"
export PYTHONPATH="${MIXSORT_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
# This host's A100 topology hangs NCCL peer-to-peer collectives.  Shared-memory
# NCCL remains functional and lets all eight cards train concurrently.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

test -s "${DATA_ROOT}/SPLIT_MANIFEST.json"
test -s "${DATA_ROOT}/annotations/clean_train.json"
test -s "${INIT_CKPT}"
mkdir -p "${OUTPUT_ROOT}"

cd "${MIXSORT_ROOT}"
TRAIN_ARGS=(
  tools/train.py
  -expn mcbyte_yolox_x_loo19
  -f "${PROJECT_ROOT}/configs/mcbyte_yolox_x_loo19.py"
  -d "${GPU_COUNT}"
  -b "${BATCH_SIZE}"
  --fp16
  -c "${INIT_CKPT}"
  "$@"
)
if [[ "${GPU_COUNT}" -gt 1 ]]; then
  # torch.distributed.run propagates child stderr and rank environment cleanly;
  # the MixSort launcher otherwise starts eight bare ``python3`` processes.
  exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${GPU_COUNT}" "${TRAIN_ARGS[@]}"
else
  exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    "${PYTHON_BIN}" "${TRAIN_ARGS[@]}"
fi
