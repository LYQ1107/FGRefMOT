#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-8}
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

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
OUT_ROOT="$ROOT/tam_vis_output_review_r2_decoder_query"
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

run_one "decoder_top1_box" --visual_source top1_box --tam_fpn_layer 1
run_one "decoder_gt_query" --visual_source gt_query --tam_fpn_layer 1
run_one "decoder_refer_response" --visual_source refer --tam_fpn_layer 1

cat > "$OUT_ROOT/README_results.txt" <<EOF
Generated at: $(date '+%F %T')
Baseline checkpoint: $BASELINE_CKPT
Ours checkpoint: $OURS_CKPT
Dataset: $RMOT_PATH

This batch targets reviewer-facing visualizations where Ours should align with GT boxes:
- decoder_top1_box: most language-confident decoder/refer query response
- decoder_gt_query: GT-aligned decoder query diagnostic response
- decoder_refer_response: sum of top refer-score query responses

Each folder contains tam_attention_comparison.png/pdf and per-sample panels/metrics.
EOF

echo "Decoder/query visualization jobs finished. Results saved to: $OUT_ROOT"
