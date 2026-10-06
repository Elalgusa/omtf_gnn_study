#!/usr/bin/env bash
# Build TPS cache for DAS external validation datasets.
# Appends to cache_das_tps (separate from the G-dataset cache_v2_tps).
# Pass --max-files N for a smoke test (e.g. --max-files 5).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
start=$(date +%s)
exec "$ROOT_DIR/venv/bin/python" -u scripts/make_dataset.py \
    --data-dir   /lhome/ext/uovi156/uovi1562/gnn_dse_hls/data/das_prod \
    --output-dir build/omtf_gmt/cache_das_tps \
    --datasets   single_muon_flatpt displaced_lowpt displaced_midpt \
                 dy_prompt llp_addon minbias \
    "$@"
end=$(date +%s)
echo "Total time: $((end - start)) seconds"