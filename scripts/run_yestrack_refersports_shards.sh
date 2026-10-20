#!/usr/bin/env bash
set -euo pipefail

# Run the leakage-safe YesTrack evaluation on disjoint video shards. The
# model is read from the local cache once per process; tracker files remain the
# shared McByte outputs on NAS and are never generated from ground truth.
PROJECT="${PROJECT:-/data3/liuyeqiang/FGRefMOT_RMOT_Sports_20261019}"
PYTHON="${PYTHON:-/data3/liuyeqiang/.venvs/fgrefmot-rmot-sports-20261019/bin/python}"
MODEL="${MODEL:-/data3/liuyeqiang/.cache/fgrefmot/Qwen3-VL-2B-Instruct}"
LORA="${LORA:-$PROJECT/runs/gpu/yestrack_y1_sports_lora_seed42_v2/runs/20261019_184604/train/lora_epoch_0}"
SHARDS="${SHARDS:-$PROJECT/runs/yestrack_eval_mcb_20261020/expression_shards}"
TRACKER="${TRACKER:-/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/assets/pretrained/common_tracker/mcbyte_sportsmot_loo19_fast5_20261020/track_results}"
TRACKER_SHARDS="${TRACKER_SHARDS:-$PROJECT/runs/yestrack_eval_mcb_20261020/tracker_shards}"
BASE="${BASE:-/data3/liuyeqiang/nas/A100/FGREF_RMOT_SPORTS_20261019/runs/gpu/yestrack_y1_mcbyte_parallel_20261020}"
LOG_DIR="${LOG_DIR:-$PROJECT/runs/yestrack_eval_mcb_20261020/logs}"
SHARD_COUNT="${SHARD_COUNT:-32}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"

# Qwen3-VL creates a large CPU thread pool by default.  With many independent
# GPU workers that silently oversubscribes the 40-core host and makes the
# referring stage slower.  Keep BLAS/PyTorch at one thread per worker; image
# crops still use the evaluator's small preprocessing pool.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

[[ -x "$PYTHON" ]] || { echo "missing Python: $PYTHON" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "missing local model: $MODEL" >&2; exit 2; }
[[ -d "$LORA" ]] || { echo "missing LoRA adapter: $LORA" >&2; exit 2; }
[[ -d "$TRACKER" ]] || { echo "missing tracker root: $TRACKER" >&2; exit 2; }

if [[ "$RESET_OUTPUTS" == "1" ]]; then
    rm -rf "$BASE" "$TRACKER_SHARDS"
fi
mkdir -p "$BASE" "$TRACKER_SHARDS" "$LOG_DIR"
pids=()
for ((i = 0; i < SHARD_COUNT; i++)); do
    shard=$(printf '%02d' "$i")
    # Four independent referring workers per A100 keep the GPU busy while each
    # worker processes a small number of expression units.
    gpu=$((i % 8 + 1))
    data_root="$SHARDS/val_$shard"
    tracker_root="$TRACKER_SHARDS/val_$shard"
    out_root="$BASE/shard_$shard"
    mkdir -p "$tracker_root"
    while IFS= read -r sequence; do
        [[ -z "$sequence" || "$sequence" == \#* ]] && continue
        [[ -f "$TRACKER/$sequence.txt" ]] || { echo "missing tracker file: $TRACKER/$sequence.txt" >&2; exit 2; }
        if [[ ! -e "$tracker_root/$sequence.txt" ]]; then
            ln -s "$TRACKER/$sequence.txt" "$tracker_root/$sequence.txt"
        fi
    done < "$data_root/sequence_list.txt"
    "$PYTHON" "$PROJECT/scripts/eval_yestrack_refersports.py" \
        --python "$PYTHON" \
        --model-name "$MODEL" \
        --lora-path "$LORA" \
        --data-root "$data_root" \
        --tracker-root "$tracker_root" \
        --results-root "$out_root/results" \
        --gt-manifest-root "$PROJECT/runs/refer_sports_mot_gt_test_v2" \
        --output-root "$out_root" \
        --sequence-list "$data_root/sequence_list.txt" \
        --gpu "$gpu" \
        --infer-every-n-frames 1 \
        --thresholds 0.5 \
        --run-inference \
        --inference-only \
        --json >"$LOG_DIR/yestrack_mcb_shard_$shard.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
