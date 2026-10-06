#!/usr/bin/env bash
# G9/G10 high-eta hard-neg retraining: Run A — w_hard_neg=0.25, lr=5e-4, G10×1
# G10 is large (~430k samples) so ×4 dominated 28% of the epoch; capped at ×1.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/venv/bin/python"
OUTDIR="/lhome/ext/uovi156/uovi1564/GNN/omtf_gnn_study/build/omtf_gmt/checkpoints/two_models/edge_transf2_edges_h64_tps_g9g10_runa"
EVALDIR="/lhome/ext/uovi156/uovi1564/GNN/omtf_gnn_study/build/omtf_gmt/eval/runa/two_models/edge_transf2_edges"


startt=$(date +%s)
echo "=== TRAIN edge_transf2_edges h64 TPS g9g10 Run A (w_hard_neg=0.25) ==="
"$PYTHON" -u src/omtf_gmt/train.py \
    --cache-dir  /lhome/ext/uovi156/uovi1562/gnn_dse_hls/build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4 \
    --repeat     B4:4 G7:4 G8:3 G9:4 G10:1 \
    --model      edge_transf2_edges \
    --hidden     64 \
    --epochs     100 \
    --lr         5e-4 \
    --batch-size 4096 \
    --num-workers 4 \
    --amp \
    --scheduler  cosine \
    --save-epochs 25 50 75 100 \
    --w-hard-neg 0.25 \
    --output-dir "$OUTDIR" \
    --device     cuda
endt=$(date +%s)
echo "Total time (Training): $((endt - startt)) seconds"

startr=$(date +%s)
echo "=== TRAIN Regression h64 TPS g9g10 Run A (w_hard_neg=0.25) ==="
"$PYTHON" -u src/omtf_gmt/train_regression.py \
    --cache-dir       /lhome/ext/uovi156/uovi1562/gnn_dse_hls/build/omtf_gmt/cache_v2_tps \
    --datasets        G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4 \
    --repeat          B4:4 G7:4 G8:3 G9:4 G10:1 \
    --classifier-ckpt "$OUTDIR/gmt_edge_transf2_edges_best.pt" \
    --model           regress_pt_charge \
    --hidden          64 \
    --epochs          100 \
    --batch-size      4096 \
    --w-pt            0.5 \
    --lr              5e-4 \
    --num-workers     4 \
    --amp \
    --scheduler       cosine \
    --save-epochs     25 50 75 100 \
    --output-dir      "$OUTDIR" \
    --device          cuda
endr=$(date +%s)
echo "Total time (Regression): $((endr - startr)) seconds"

starte=$(date +%s)
echo "=== EVAL best ==="
"$PYTHON" -u scripts/eval.py \
    --classifier-ckpt "$OUTDIR/gmt_edge_transf2_edges_best.pt" \
    --regression-ckpt "$OUTDIR/gmt_regress_pt_charge_best.pt" \
    --cache-dir  /lhome/ext/uovi156/uovi1562/gnn_dse_hls/build/omtf_gmt/cache_v2_tps \
    --datasets   G1 G2 G3 G4 G5 G6 G7 G8 G9 G10 B4 \
    --threshold  0.0 \
    --output     "$EVALDIR/edge_transf2_edges_h64_tps_g9g10_runa_best_eval.md" \
    --device     cuda
ende=$(date +%s)
echo "Total time (Evaluation): $((ende - starte)) seconds"
