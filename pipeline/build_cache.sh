#!/usr/bin/env bash
# Build full GMT-TPS cache for Phase B4 (all datasets, all files).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
start=$(date +%s)
exec "$ROOT_DIR/venv/bin/python" -u scripts/make_dataset.py \
    --data-dir   data/prod \
    --output-dir build/omtf_gmt/cache_v2_tps \
    --datasets   G1_pos G1_neg G2_pos G2_neg G3_pos G3_neg \
                 G4_pos G4_neg G5_pos G5_neg G6_pos G6_neg \
                 G7 G8 B4 \
    "$@"
end=$(date +%s)
echo "Total time: $((end - start)) seconds"