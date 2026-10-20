#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019/bin/python}"
DATA_ROOT="${RMOT_MCBYTE_TRACKER_DATA:-/data3/liuyeqiang/datasets/SportsMOT/sportsmot_publish/dataset}"
CHECKPOINT="${RMOT_MCBYTE_TRACKER_CHECKPOINT:-${PROJECT_ROOT}/runs/gpu/mcbyte_yolox_x_fast5_20261020/mcbyte_yolox_x_loo19/latest_ckpt.pth.tar}"
OUTPUT_ROOT="${RMOT_MCBYTE_TRACKER_OUTPUT:-/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_sportsmot_loo19_fast5_20261020/track_results}"
SEQUENCE_FILE="${RMOT_MCBYTE_TRACKER_SEQUENCE_FILE:-${PROJECT_ROOT}/configs/refer_sports_eval_sequences_20261019.txt}"
GPU_LIST="${RMOT_MCBYTE_TRACKER_GPUS:-1,2,3,4,5,6,7,8}"

test -s "${CHECKPOINT}"
test -s "${SEQUENCE_FILE}"
mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

mapfile -t SEQUENCES < <(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "${SEQUENCE_FILE}")
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
(( ${#GPUS[@]} > 0 ))

PIDS=()
for idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$idx]}"
  worker_sequences=()
  for seq_idx in "${!SEQUENCES[@]}"; do
    if (( seq_idx % ${#GPUS[@]} == idx )); then
      worker_sequences+=("${SEQUENCES[$seq_idx]}")
    fi
  done
  ((${#worker_sequences[@]} > 0)) || continue
  log_file="${OUTPUT_ROOT}/logs/worker_gpu${gpu}.log"
  echo "GPU ${gpu}: ${worker_sequences[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}/third_party/MixSort:${PROJECT_ROOT}:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/mcbyte_refersports_tracker.py" \
    --exp-file "${PROJECT_ROOT}/configs/mcbyte_yolox_x_loo19.py" \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --sequences "${worker_sequences[@]}" \
    --test-size "${RMOT_MCBYTE_TRACKER_TEST_SIZE:-512x896}" \
    --batch-size "${RMOT_MCBYTE_TRACKER_BATCH_SIZE:-8}" \
    --conf "${RMOT_MCBYTE_TRACKER_CONF:-0.10}" \
    --nms "${RMOT_MCBYTE_TRACKER_NMS:-0.70}" \
    --track-thresh "${RMOT_MCBYTE_TRACKER_THRESH:-0.50}" \
    --track-buffer "${RMOT_MCBYTE_TRACKER_BUFFER:-30}" \
    --match-thresh "${RMOT_MCBYTE_TRACKER_MATCH:-0.80}" \
    --min-box-area "${RMOT_MCBYTE_TRACKER_MIN_AREA:-100}" \
    --fp16 \
    >"${log_file}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=$?
done
if [[ "${status}" -eq 0 ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/assemble_mcbyte_tracker_manifest.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --checkpoint "${CHECKPOINT}" \
    --sequence-file "${SEQUENCE_FILE}" \
    --test-size "${RMOT_MCBYTE_TRACKER_TEST_SIZE:-512x896}" \
    --conf "${RMOT_MCBYTE_TRACKER_CONF:-0.10}" \
    --nms "${RMOT_MCBYTE_TRACKER_NMS:-0.70}" \
    --track-thresh "${RMOT_MCBYTE_TRACKER_THRESH:-0.50}" \
    --track-buffer "${RMOT_MCBYTE_TRACKER_BUFFER:-30}" \
    --match-thresh "${RMOT_MCBYTE_TRACKER_MATCH:-0.80}"
fi
exit "${status}"
