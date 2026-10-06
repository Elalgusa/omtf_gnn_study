#!/usr/bin/env bash
# Append G9/G10 high-eta hard-negative datasets to the existing TPS cache.
# The builder merges G9/G10 into the manifest without touching existing shards.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/venv/bin/python" -u scripts/make_dataset.py \
    --data-dir   data/prod \
    --output-dir build/omtf_gmt/cache_v2_tps \
    --datasets   G9_pos G9_neg G10_pos G10_neg \
    "$@"
