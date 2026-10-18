#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8}

if [ -f "/home/liuyeqiang/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/liuyeqiang/anaconda3/etc/profile.d/conda.sh"
  conda activate mamMOT
fi

ROOT="/data1/liuyeqiang/TempMam"
PYTHON_BIN="${PYTHON_BIN:-python}"
VIS_SCRIPT="$ROOT/visualize_tam_attention.py"
BASELINE_CKPT="$ROOT/exps/default_debug_rk/checkpoint0049.pth"
OURS_CKPT="$ROOT/exps/default_debug_rk/checkpoint0089.pth"
RMOT_PATH="$ROOT/data/Dataset/refer-kitti"
OUT_ROOT="$ROOT/tam_vis_output_review_r2_all"
LOG_DIR="$OUT_ROOT/logs"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --baseline_ckpt "$BASELINE_CKPT"
  --ours_ckpt "$OURS_CKPT"
  --rmot_path "$RMOT_PATH"
  --search_frames
  --save_all
  --warmup_frames 15
  --overlay_style gradcam
)

run_one() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  local log_file="$LOG_DIR/$name.log"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] Start $name" | tee "$log_file"
  "$PYTHON_BIN" "$VIS_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --vis_output_dir "$out_dir" \
    "$@" 2>&1 | tee -a "$log_file"
  echo "[$(date '+%F %T')] Done $name" | tee -a "$log_file"
}

run_one "tam_layer1" --visual_source tam --tam_fpn_layer 1
run_one "tam_layer2" --visual_source tam --tam_fpn_layer 2
run_one "tam_layer3" --visual_source tam --tam_fpn_layer 3
run_one "refer_response_layer1" --visual_source refer --tam_fpn_layer 1

cat > "$OUT_ROOT/README_results.txt" <<EOF
Generated at: $(date '+%F %T')
Baseline checkpoint: $BASELINE_CKPT
Ours checkpoint: $OURS_CKPT
Dataset: $RMOT_PATH

Folders:
- tam_layer1: true TAM text-to-visual attention, FPN layer 1
- tam_layer2: true TAM text-to-visual attention, FPN layer 2
- tam_layer3: true TAM text-to-visual attention, FPN layer 3
- refer_response_layer1: decoder refer-score response-map backup

Each folder contains:
- tam_attention_comparison.png/pdf
- samples/<sample_id>/input_gt.png
- samples/<sample_id>/baseline_tam_overlay.png
- samples/<sample_id>/ours_tam_overlay.png
- samples/<sample_id>/baseline_tam_heat.npy
- samples/<sample_id>/ours_tam_heat.npy
- samples/<sample_id>/metrics.json
- logs are in logs/*.log
EOF

echo "All visualization jobs finished. Results saved to: $OUT_ROOT"
