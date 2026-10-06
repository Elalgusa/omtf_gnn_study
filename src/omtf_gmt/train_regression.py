## # INCLUIR introducción/explicación
"""
Sistema de comentarios:
###   Cosas por hacer
## #  Cosas por hacer (menos prioridad)
## ## Sugerencias
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from omtf_gmt.dataset import GMTCachedDataset, collate_gmt, expand_datasets, expand_repeats
from omtf_gmt.models_regression import build_regress_pt_charge
from omtf_gmt.models  import build_deepsets, build_edge_compat, build_edge_compat_assign, build_edge_compat_edges, build_edge_transf_edges, build_edge_transf2_edges, build_slot_model, build_seq_slot, build_count_model, build_detr_model
from omtf_gmt.models.slot_model import (
    candidate_count_loss,
    attention_diversity_loss,
    null_attention_empty_slot_loss,
    attention_supervision_loss,
)

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

K_MAX          = 3
ALL_DATASETS   = ["S1", "S2", "S3", "S4", "S5", "B1", "B2", "B3", "B4"]
ALL_G_DATASETS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "B4"]

# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #

def compute_loss(
    out_regression:    dict[str, torch.Tensor],
    batch:  dict[str, torch.Tensor],
    device,
    w_pt:             float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    gpt         = batch["gen_pt"]                  # (N, K)
    cand_target = (gpt > 0).float()

    # pT regression (log-space MSE, signal slots only)
    sig_mask = cand_target > 0.5
    pt_loss  = torch.tensor(0.0, device=device)
    if sig_mask.any():
        pt_pred_pos = out_regression["pt_pred"]   # models apply F.softplus internally
        pt_loss = F.mse_loss(
            torch.log1p(pt_pred_pos[sig_mask]),
            torch.log1p(gpt[sig_mask]),
        )
    
    ## # INCLUIR loss charge_pred

    total = w_pt * pt_loss
    breakdown = {
        "pt_loss":   pt_loss.item(),
    }

    breakdown["loss"] = total.item()
    return total, breakdown

# --------------------------------------------------------------------------- #
# Metrics (simple, for smoke-test)
# --------------------------------------------------------------------------- #

## # MEJORAR el quick.metrics para regression
@torch.no_grad()
def quick_metrics(out: dict, batch: dict) -> dict[str, float]:
    gpt = batch["gen_pt"]
    ppt = out["pt_pred"]
    mean_abs_error    = torch.mean(torch.abs(gpt - ppt)).item()
    r_mean_sqrt_error = torch.sqrt(torch.mean((gpt - ppt) ** 2)).item()
    
    ss_res = torch.sum((gpt - ppt) ** 2)
    ss_tot = torch.sum((gpt - torch.mean(gpt)) ** 2)
    r2     = (1 - ss_res / ss_tot).item()
    return {"mean_absolute_error": mean_abs_error, "r_mean_squared_error": r_mean_sqrt_error, "r2_score": r2}

### INCLUIR un quick metrics para charge regression

# --------------------------------------------------------------------------- #
# Load Classification Model
# --------------------------------------------------------------------------- #


def load_model(ckpt_path: Path, device: torch.device):
    """
    Load classification model from checkpoint.

    The checkpoint is expected to contain:
      - "model": state_dict
      - "args": training args, including model name, hidden dim, dropout
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})

    mname = args.get("model", "deepsets")
    hdim = int(args.get("hidden", 64))
    dropout = float(args.get("dropout", 0.0))

    if mname == "deepsets":
        model = build_deepsets(hidden=hdim, dropout=dropout)
    elif mname == "edge_compat":
        model = build_edge_compat(hidden=hdim, dropout=dropout)
    elif mname == "edge_compat_assign":
        model = build_edge_compat_assign(hidden=hdim, dropout=dropout)
    elif mname == "edge_compat_edges":
        model = build_edge_compat_edges(hidden=hdim, dropout=dropout)
    elif mname == "edge_transf_edges":
        model = build_edge_transf_edges(hidden=hdim, dropout=dropout)
    elif mname == "edge_transf2_edges":
        model = build_edge_transf2_edges(hidden=hdim, dropout=dropout)
    elif mname == "slot_model":
        model = build_slot_model(hidden=hdim, dropout=dropout)
    elif mname == "seq_slot":
        model = build_seq_slot(hidden=hdim, dropout=dropout)
    elif mname == "count_model":
        model = build_count_model(hidden=hdim, dropout=dropout)
    elif mname == "detr_model":
        model = build_detr_model(hidden=hdim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type in checkpoint: {mname!r}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    return (
        model,
        mname,
        hdim,
        dropout,
        ckpt.get("epoch", "?"),
        ckpt.get("best_val_loss", float("nan")),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GMT branch training (Regression)")
    p.add_argument("--cache-dir",             type=Path, required=True)
    p.add_argument("--classifier-ckpt", type=Path, required=True)
    p.add_argument("--datasets",     nargs="+", default=ALL_DATASETS)
    p.add_argument("--repeat",       nargs="*", default=["B4:8"], metavar="DS:N",
                   help="per-dataset train-split repeat factors, e.g. --repeat B4:8 S4:2")
    p.add_argument("--model",        default="regress_pt_charge",
                   choices=["regress_pt_charge"])
    p.add_argument("--hidden",       type=int,   default=64)
    p.add_argument("--dropout",      type=float, default=0.0)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=512)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--w-pt",         type=float, default=0.5)
    p.add_argument("--num-workers",  type=int,   default=4,
                   help="DataLoader worker processes (0 = main process only)")
    p.add_argument("--amp",          action="store_true", default=False,
                   help="enable automatic mixed precision (fp16) — recommended on V100+")
    p.add_argument("--scheduler",    default="none", choices=["none", "cosine"],
                   help="LR scheduler: cosine annealing (eta_min = lr * 0.05)")
    p.add_argument("--save-epochs",  nargs="*", type=int, default=[],
                   help="extra epochs to save checkpoints at, e.g. --save-epochs 50 75 100")
    p.add_argument("--output-dir",   type=Path,  default=Path("build/omtf_gmt/checkpoints_regression"))
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    counter = 0
    args   = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark        = True  # autotune kernels for fixed input shapes
        torch.backends.cuda.matmul.allow_tf32 = True  # A100 TF32 tensor cores for fp32 matmuls
        torch.backends.cudnn.allow_tf32       = True

    # --- classification model ---
    classifier, classifier_name, classifier_hid, classifier_dropout, classifier_epoch, classifier_best_loss = load_model(args.classifier_ckpt, device)
    classifier_params = sum(p.numel() for p in classifier.parameters())
    print(
        f"Classifier Model: {classifier_name} hidden={classifier_hid} dropout={classifier_dropout} "
        f"params={classifier_params:,} epoch={classifier_epoch} best_val_loss={classifier_best_loss:.4f}"
    )

    # --- parse per-dataset repeat factors (alias-aware) ---
    repeats = expand_repeats(args.repeat)

    # --- data ---
    train_parts: list = []
    val_parts:   list = []
    ds_sizes: list[tuple[str, int, int]] = []   # (name, n_train_1x, n_repeat)

    for ds in expand_datasets(args.datasets):
        full  = GMTCachedDataset(args.cache_dir, ds)
        n     = len(full)
        n_tr  = int(0.85 * n)
        n_va  = n - n_tr
        tr, va = torch.utils.data.random_split(
            full, [n_tr, n_va],
            generator=torch.Generator().manual_seed(42),
        )
        train_parts.append(tr)
        val_parts.append(va)
        ds_sizes.append((ds, n_tr, repeats.get(ds, 1)))

    ### REVISAR esta parte
    # WeightedRandomSampler: each dataset's samples get weight = repeat factor.
    # Draws are independent with replacement, so high-weight datasets appear
    # more often but in random positions throughout the epoch — no block patterns.
    sample_weights: list[float] = []
    for _, n_tr, n_rep in ds_sizes:
        sample_weights.extend([float(n_rep)] * n_tr)

    n_train_eff = sum(n_tr * n_rep for _, n_tr, n_rep in ds_sizes)
    n_val_total = sum(len(d) for d in val_parts)

    pin = device.type == "cuda"
    pw  = args.num_workers > 0

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=n_train_eff,
        replacement=True,
    )
    train_loader = DataLoader(
        ConcatDataset(train_parts), batch_size=args.batch_size,
        sampler=sampler, collate_fn=collate_gmt,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw,
    )
    val_loader = DataLoader(
        ConcatDataset(val_parts), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_gmt,
        num_workers=args.num_workers, pin_memory=pin, persistent_workers=pw,
    )

    print("Effective training mix (WeightedRandomSampler):")
    for ds_name, n_tr, n_rep in ds_sizes:
        eff = n_tr * n_rep
        pct = 100.0 * eff / max(1, n_train_eff)
        rep_str = f" ×{n_rep}" if n_rep > 1 else ""
        print(f"  {ds_name}{rep_str}: {eff:,} ({pct:.1f}%)")
    print(f"  Total train: {n_train_eff:,}  |  Val: {n_val_total:,}  |  Device: {device}")
    # Note: n_train_eff > raw_train_size when any repeat > 1 — epoch is proportionally longer.

    mix_info = {
        "datasets": [
            {
                "name": ds, "n_train_1x": n_tr, "repeat": n_rep,
                "effective": n_tr * n_rep,
                "fraction": (n_tr * n_rep) / max(1, n_train_eff),
            }
            for ds, n_tr, n_rep in ds_sizes
        ],
        "n_train_eff": n_train_eff,
        "n_val_total": n_val_total,
        "sampler": "WeightedRandomSampler(replacement=True)",
        "amp": args.amp,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
    }
    (args.output_dir / "training_mix.json").write_text(json.dumps(mix_info, indent=2))

    # --- model ---
    if args.model == "regress_pt_charge":
        model = build_regress_pt_charge(hidden=args.hidden, dropout=args.dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  params={n_params:,}")

    opt    = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.05
        )
        print(f"Scheduler: CosineAnnealingLR  T_max={args.epochs}  eta_min={args.lr * 0.05:.2e}")

    save_epochs = set(args.save_epochs)
    history = []
    best_val_loss = float("inf")
    ckpt_path = args.output_dir / f"gmt_{args.model}_best.pt"
    last_path = args.output_dir / f"gmt_{args.model}_last.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # train
        model.train()
        tr_loss    = 0.0
        ## ## USAR tr_hn_loss y hn_str como ejemplo para un training error for displaced
        for batch in train_loader:
            if counter == 0:
                print("Batch type:", type(batch))
                print("Batch keys:", batch.keys())
            batch = {k: v.to(device, non_blocking=pin) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            # classifier evaluation
            stubs = batch["stubs"]
            vm = batch["valid_mask"]
            N, Nmax, _ = stubs.shape
            ndevice     = stubs.device
            if classifier_name == "edge_compat_edges":
                batch["edge_node1"] = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1)     # (N, Nmax, Nmax, F)
                batch["edge_node2"] = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1)     # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                        # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool,
                                        device=device).unsqueeze(0)                   # (1, Nmax, Nmax)
                batch["edge_mask"]  = pair_valid & no_self                            # (N, Nmax, Nmax)
            elif (classifier_name == "edge_transf_edges" or classifier_name == "edge_transf2_edges"):
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1)              # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1)              # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                        # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool,
                                        device=device).unsqueeze(0)                   # (1, Nmax, Nmax)
                edge_mask  = pair_valid & no_self                                     # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)    # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                batch["edge_index"] = torch.stack([src_node, dst_node], dim=0)        # (2, E)
                batch["edge_attr"]  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,)                                   # (E, 2 * F)
            if classifier_name == "edge_compat_edges":
                classifier_out = classifier(batch["stubs"], batch["valid_mask"], batch["edge_node1"], batch["edge_node2"], batch["edge_mask"])
            elif (classifier_name == "edge_transf_edges" or classifier_name == "edge_transf2_edges"):
                classifier_out = classifier(batch["stubs"], batch["valid_mask"], batch["edge_index"], batch["edge_attr"])
            else:
                classifier_out = classifier(batch["stubs"], batch["valid_mask"])
            
            if args.model == "regress_pt_charge":
                batch["global_ctx"] = classifier_out["global_ctx"]
            
            if counter == 0:
                print("Batch type:", type(batch))
                print("Batch keys:", batch.keys())
                counter += 1

            with torch.amp.autocast('cuda', enabled=args.amp):
                if args.model == "regress_pt_charge":
                    out = model(batch["global_ctx"])
                if args.model == "regress_pt_charge":
                    loss, bd = compute_loss(out, batch, args.device, args.w_pt)
            if counter == 0:
                counter += 1
                print("out type", type(out))
                print("out keys", out.keys())
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tr_loss    += loss.item()
        if scheduler is not None:
            scheduler.step()
        tr_loss    /= max(1, len(train_loader))

        # validate
        model.eval()
        val_loss = 0.0
        
        val_metrics: dict[str, float] = {"mean_absolute_error": 0, "r_mean_squared_error": 0, "r2_score": 0 } ### INCLUIR metrics
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device, non_blocking=pin) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                
                # classifier evaluation
                stubs = batch["stubs"]
                vm = batch["valid_mask"]
                N, Nmax, _ = stubs.shape
                device     = stubs.device
                if classifier_name == "edge_compat_edges":
                    batch["edge_node1"] = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1)     # (N, Nmax, Nmax, F)
                    batch["edge_node2"] = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1)     # (N, Nmax, Nmax, F)
                    pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                        # (N, Nmax, Nmax)
                    no_self    = ~torch.eye(Nmax, dtype=torch.bool,
                                            device=device).unsqueeze(0)                   # (1, Nmax, Nmax)
                    batch["edge_mask"]  = pair_valid & no_self                            # (N, Nmax, Nmax)
                elif (classifier_name == "edge_transf_edges" or classifier_name == "edge_transf2_edges"):
                    edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1)              # (N, Nmax, Nmax, F)
                    edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1)              # (N, Nmax, Nmax, F)
                    pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                        # (N, Nmax, Nmax)
                    no_self    = ~torch.eye(Nmax, dtype=torch.bool,
                                            device=device).unsqueeze(0)                   # (1, Nmax, Nmax)
                    edge_mask  = pair_valid & no_self                                     # (N, Nmax, Nmax)
                    graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)    # (E,)
                    src_node = graph_idx * Nmax + node1_idx
                    dst_node = graph_idx * Nmax + node2_idx
                    batch["edge_index"] = torch.stack([src_node, dst_node], dim=0)        # (2, E)
                    batch["edge_attr"]  = torch.cat([
                                            edge_node1[graph_idx, node1_idx, node2_idx],
                                            edge_node2[graph_idx, node1_idx, node2_idx],
                                            ], dim=-1,)                                   # (E, 2 * F)
                if classifier_name == "edge_compat_edges":
                    classifier_out = classifier(batch["stubs"], batch["valid_mask"], batch["edge_node1"], batch["edge_node2"], batch["edge_mask"])
                elif (classifier_name == "edge_transf_edges" or classifier_name == "edge_transf2_edges"):
                    classifier_out = classifier(batch["stubs"], batch["valid_mask"], batch["edge_index"], batch["edge_attr"])
                else:
                    classifier_out = classifier(batch["stubs"], batch["valid_mask"])
                
                if args.model == "regress_pt_charge":
                    batch["global_ctx"] = classifier_out["global_ctx"]

                with torch.amp.autocast('cuda', enabled=args.amp):
                    if args.model == "regress_pt_charge":
                        out = model(batch["global_ctx"])
                if args.model == "regress_pt_charge":
                    loss, bd = compute_loss(out, batch, args.device, args.w_pt)
                val_loss += loss.item()
                m = quick_metrics(out, batch)
                for k in val_metrics:
                    val_metrics[k] += m[k]
        val_loss /= max(1, len(val_loader))
        for k in val_metrics:
            val_metrics[k] /= max(1, len(val_loader))

        elapsed = time.time() - t0
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"tr={tr_loss:.4f}  val={val_loss:.4f} "
            f"({elapsed:.0f}s)"
        )

        row = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss, **val_metrics}
        history.append(row)

        ckpt_payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            torch.save(ckpt_payload, ckpt_path)

        # always overwrite last.pt
        torch.save(ckpt_payload, last_path)

        # fixed-epoch snapshots
        if epoch in save_epochs:
            snap_path = args.output_dir / f"gmt_{args.model}_epoch_{epoch:04d}.pt"
            torch.save(ckpt_payload, snap_path)
            print(f"  Snapshot saved: {snap_path}")

    # save history
    hist_path = args.output_dir / f"gmt_{args.model}_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"\nBest val_loss={best_val_loss:.4f} at checkpoint: {ckpt_path}")
    print(f"History: {hist_path}")


if __name__ == "__main__":
    main()
