#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
start=$(date +%s)
exec "$ROOT_DIR/venv/bin/python" -u scripts/make_das_plots_reduced.py \
    --checkpoint   build/omtf_gmt/checkpoints/frozen_fp32_tps_edgetransf2edges_h64/model.pt \
    --cache-dir    build/omtf_gmt/cache_das_tps \
    --das-prod-dir /lhome/ext/uovi156/uovi1562/gnn_dse_hls/data/das_prod \
    --unfiltered   /lhome/ext/uovi156/uovi1562/gnn_dse_hls/build/omtf_gmt/eval/frozen_fp32_das_eval.json \
    --filtered     build/omtf_gmt/eval/runa/edge_transf2_edges/frozen_fp32_das_filtered_eval.json \
    --output-dir   build/omtf_gmt/plots/das_validation/runa/edge_transf2_edges/reduced \
    --device       cuda \
    "$@"
end=$(date +%s)
echo "Total time: $((end - start)) seconds"