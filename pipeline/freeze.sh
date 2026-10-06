#!/usr/bin/env bash
# Freeze the best FP32 TPS EdgeCompat h64 checkpoint (Run A, G9/G10-trained).
# Creates build/omtf_gmt/checkpoints/frozen_fp32_tps_edgecompat_h64/ with all
# artefacts needed for QAT, HLS, and external DAS validation.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"

SRC="/lhome/ext/uovi156/uovi1564/PelayoGNN/omtf_gnn_study/build/omtf_gmt/checkpoints/check/edge_transf2_edges_h64_tps_g9g10_runa"
EVAL_SRC="/lhome/ext/uovi156/uovi1564/PelayoGNN/omtf_gnn_study/build/omtf_gmt/eval/runa/check/edge_transf2_edges/edge_transf2_edges_h64_tps_g9g10_runa_best_eval"
FROZEN="build/omtf_gmt/checkpoints/check/frozen_fp32_tps_edgetransf2edges_h64"

mkdir -p "$FROZEN"
start=$(date +%s)
echo "Copying model and training artefacts..."
cp "$SRC/gmt_edge_transf2_edges_best.pt"   "$FROZEN/model.pt"
cp "$SRC/training_mix.json"          "$FROZEN/training_mix.json"
cp "$SRC/gmt_edge_transf2_edges_history.json" "$FROZEN/history.json"
cp "${EVAL_SRC}.md"                  "$FROZEN/eval.md"
cp "${EVAL_SRC}.json"                "$FROZEN/eval.json"

echo "Generating feature_config.json..."
"$PYTHON" - <<'PY'
import json, sys
sys.path.insert(0, "src")
from omtf_gmt.features_tps import FEATURE_NAMES_TPS, N_FEATURES_TPS
cfg = {
    "stub_source":    "MuonStubTps",
    "n_features":     N_FEATURES_TPS,
    "feature_names":  FEATURE_NAMES_TPS,
    "nmax":           24,
    "k_max":          3,
}
json.dump(cfg, open("build/omtf_gmt/checkpoints/check/frozen_fp32_tps_edgetransf2edges_h64/feature_config.json","w"), indent=2)
print("  feature_config.json written")
PY

echo "Copying cache manifest as dataset_manifest.json..."
cp "/lhome/ext/uovi156/uovi1562/gnn_dse_hls/build/omtf_gmt/cache_v2_tps/manifest.json" "$FROZEN/dataset_manifest.json"

echo "Extracting threshold_scan.json from eval.json..."
"$PYTHON" - <<'PY'
import json
ev = json.load(open("build/omtf_gmt/eval/runa/check/edge_transf2_edges/edge_transf2_edges_h64_tps_g9g10_runa_best_eval.json"))
# Collect per-dataset threshold scans from per_window entries
scan = {}
for entry in ev.get("per_window", []):
    ds = entry["ds"]
    scan[ds] = {
        "zero_threshold_scan": entry.get("zero_threshold_scan", []),
        "roc":                 entry.get("roc", []),
    }
json.dump(scan, open("build/omtf_gmt/checkpoints/check/frozen_fp32_tps_edgetransf2edges_h64/threshold_scan.json","w"), indent=2)
print("  threshold_scan.json written")
PY

echo "Writing README.md..."
cat > "$FROZEN/README.md" <<'EOF'
# Frozen FP32 checkpoint: TPS EdgeCompat h64 (Run A, G9/G10)

## Why this checkpoint was frozen

This is the selected floating-point baseline for external DAS validation and
subsequent Quantization-Aware Training (QAT).

Selected after comparing three G9/G10 training runs:
- Run A (w_hard_neg=0.25): best trade-off — G9/G10 zero_fp ≈ 0.002–0.016, G7/G8 intact
- Run C (w_hard_neg=0.10): recovered ~1% signal efficiency but lost G7 rejection (+50% FP)
- Old hn025 baseline: no G9/G10 supervision at all

Run A was chosen: it cleanly learns high-eta rejection without sacrificing
the barrel-side (G7/G8) suppression or pure-noise (B4) rejection.

## Key metrics at logit > 0.0

| Dataset | cand_eff | zero_fp | evt_trig@10 |
|---------|----------|---------|-------------|
| G1 (clean prompt)    | 0.908 | 0.48 | 0.993 |
| G2 (prompt + PU200)  | 0.803 | 0.07 | 0.959 |
| G3 (displaced)       | 0.886 | 0.61 | 0.975 |
| G4 (displaced + PU)  | 0.789 | 0.08 | 0.936 |
| G5 (2-displaced PU)  | 0.775 | 0.11 | 0.931 |
| G6 (3-prompt PU)     | 0.944 | 0.15 | 0.993 |
| G7 (low-eta HN)      | —     | 0.038 | win_bg=0.036 |
| G8 (low-eta HN + PU) | —     | 0.021 | win_bg=0.021 |
| G9 (high-eta HN)     | —     | 0.002 | win_bg=0.003 |
| G10 (high-eta HN+PU) | —     | 0.015 | win_bg=0.015 |
| B4 (pure noise PU)   | —     | 0.000 | win_bg=0.000 |

## Training configuration

- Model: EdgeCompat h64 (38,987 params)
- Cache: cache_v2_tps (TPS stubs, G1–G10 + B4)
- Mix: B4×4, G7×4, G8×3, G9×4, G10×1 + G1–G6 ×1
- LR: 5e-4 cosine, 100 epochs
- w_hard_neg: 0.25, gradient clip norm=1.0
- Best epoch: 94, best_val_loss: 0.3988
EOF

echo ""
echo "Frozen checkpoint written to: $FROZEN"
ls -lh "$FROZEN"
end=$(date +%s)
echo "Total time: $((end - start)) seconds"
