#!/usr/bin/env bash
# Evaluate the frozen FP32 model on DAS external validation datasets.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
CKPT="build/omtf_gmt/checkpoints/frozen_fp32_tps_edgecompat_h64/model.pt"
EVALDIR="build/omtf_gmt/eval"

start=$(date +%s)
echo "=== EVAL frozen model on DAS datasets ==="
"$PYTHON" -u scripts/eval.py \
    --checkpoint "$CKPT" \
    --cache-dir  build/omtf_gmt/cache_das_tps \
    --datasets   single_muon_flatpt displaced_lowpt displaced_midpt \
                 dy_prompt llp_addon minbias \
    --threshold  0.0 \
    --output     "$EVALDIR/frozen_fp32_das_eval.md" \
    --device     cuda
end=$(date +%s)
echo "Total time: $((end - start)) seconds"
