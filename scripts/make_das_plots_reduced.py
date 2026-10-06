#!/usr/bin/env python
"""
DAS validation plots — seven figures for CMS-facing validation.

Plot 1  Before/after eta-filter efficiency (bar chart, per dataset).
Plot 2  Efficiency vs gen pT (overlap-filtered signal datasets).
Plot 3  MinBias background accept vs logit threshold (window + event, pT-gated).
Plot 4  Efficiency vs |dxy| — displaced low-pT, mid-pT, LLP H→4μ (overlap-filtered).
Plot 5  Efficiency vs number of TPS stubs — same displaced datasets.
Plot 6  MinBias event accept vs predicted-pT cut at fixed logit > 0.0.
Plot 7  ML model vs current OMTF/GMT: signal efficiency and MinBias accept.

Usage:
  python scripts/omtf_gmt/make_das_validation_plots.py \\
      --checkpoint    build/omtf_gmt/checkpoints/frozen_fp32_tps_edgecompat_h64/model.pt \\
      --cache-dir     build/omtf_gmt/cache_das_tps \\
      --unfiltered    build/omtf_gmt/eval/frozen_fp32_das_eval.json \\
      --filtered      build/omtf_gmt/eval/frozen_fp32_das_filtered_eval.json \\
      --output-dir    build/omtf_gmt/plots/das_validation \\
      --device        cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("matplotlib is required.  pip install matplotlib")
    sys.exit(1)



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_DS = ["single_muon_flatpt", "displaced_lowpt", "displaced_midpt",
             "dy_prompt", "llp_addon"]
DS_LABELS = {
    "single_muon_flatpt": "Single μ (flat pT)",
    "displaced_lowpt":    "Displaced μ (low pT)",
    "displaced_midpt":    "Displaced μ (mid pT)",
    "dy_prompt":          "DY → μμ",
    "llp_addon":          "LLP H→4μ",
    "minbias":            "MinBias",
}
DS_SAMPLES = ["single_muon_flatpt", "displaced_lowpt", "displaced_midpt",
             "dy_prompt", "llp_addon", "minbias"]
PT_CUT_LABELS = {
    0:  "no pT cut",
    5:  "pred pT > 5 GeV",
    10: "pred pT > 10 GeV",
    15: "pred pT > 15 GeV",
    20: "pred pT > 20 GeV",
}
PT_BINS      = [0, 2, 5, 10, 15, 20, 30, 50, 100, 9999]
PT_TRIG_THR  = [0, 5, 10, 15, 20]
PT_CUTS      = [0, 5, 10, 15, 20]
THR_SCAN     = np.linspace(-3.0, 3.0, 121)
K_MAX        = 3

DISPLACED_DS = ["displaced_lowpt", "displaced_midpt", "llp_addon"]

DXY_BINS     = [0, 0.5, 1, 2, 5, 10, 20, 50, 300]   # cm
NSTUBS_BINS  = [1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 24]  # TPS stubs per window

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

FEAT_STUB_IN_OVERLAP = 11   # feature index: 1.0 if 0.83 <= |eta| <= 1.24


def overlap_signal_mask(stubs: torch.Tensor,
                        node_label: torch.Tensor,
                        valid_mask: torch.Tensor) -> torch.Tensor:
    """(B,) bool: window has at least one signal stub with stub_in_overlap=1."""
    sig   = (node_label > 0.5) & valid_mask
    in_ov = stubs[:, :, FEAT_STUB_IN_OVERLAP] > 0.5
    return (sig & in_ov).any(dim=1)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args  = ckpt.get("args", {})
    mname = args.get("model", "edge_compat")
    hdim  = int(args.get("hidden", 64))
    drop  = float(args.get("dropout", 0.0))
    if mname == "edge_compat":
        from omtf_gmt.models import build_edge_compat
        model = build_edge_compat(hidden=hdim, dropout=drop)
    elif mname == "edge_compat_assign":
        from omtf_gmt.models.edge_compat_assign import build_edge_compat_assign
        model = build_edge_compat_assign(hidden=hdim, dropout=drop)
    elif mname == "edge_compat_edges":
        from omtf_gmt.models.edge_compat_edges import build_edge_compat_edges
        model = build_edge_compat_edges(hidden=hdim, dropout=drop)
    elif mname == "edge_transf_edges":
        from omtf_gmt.models.edge_transf_edges import build_edge_transf_edges
        model = build_edge_transf_edges(hidden=hdim, dropout=drop)
    elif mname == "edge_transf2_edges":
        from omtf_gmt.models.edge_transf2_edges import build_edge_transf2_edges
        model = build_edge_transf2_edges(hidden=hdim, dropout=drop)
    else:
        from omtf_gmt.models import build_edge_compat
        model = build_edge_compat(hidden=hdim, dropout=drop)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model

# ---------------------------------------------------------------------------
# MinBias pT-gated scan
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_minbias_pt_scan(
    model,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 2048,
) -> dict:
    """
    Collect per-window (logits, pt_pred, event_num) for MinBias.
    Returns threshold-scan curves for window FP and event-level OR accept,
    for several predicted-pT cuts.
    """
    ds_dir = cache_dir / "minbias"
    shards = sorted(ds_dir.glob("shard_*.pt"))
    print(f"  [minbias pT scan] {len(shards)} shards ...", flush=True)

    all_logits:  list[torch.Tensor] = []
    all_ptpred:  list[torch.Tensor] = []
    all_en:      list[torch.Tensor] = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        n = int(shard["stubs"].shape[0])
        logits_l, pt_l = [], []

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            stubs = shard["stubs"][i:j].to(device)
            vm    = shard["valid_mask"][i:j].to(device)
            if model.name == "edge_compat_edges":
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                out = model(stubs, vm, edge_node1, edge_node2, edge_mask)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)             # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                edge_index = torch.stack([src_node, dst_node], dim=0).to(device)               # (2, E)
                edge_attr  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,).to(device)                                 # (E, 2 * F)
                out = model(stubs, vm, edge_index, edge_attr)
            else:
                out = model(stubs, vm)
            logits_l.append(out["candidate_logits"].cpu())
            pt_l.append(out["pt_pred"].cpu())

        all_logits.append(torch.cat(logits_l))
        all_ptpred.append(torch.cat(pt_l))
        #all_en.append(shard["meta_event_num"])

    logits_all = torch.cat(all_logits).numpy()   # (N, K)
    ptpred_all = torch.cat(all_ptpred).numpy()   # (N, K)
    en_all     = torch.cat(all_en).numpy().astype(np.int64)

    # Build event groups
    file_id = 0
    prev_en = int(en_all[0])
    event_key = np.empty(len(en_all), dtype=np.int64)
    # encode as file_id * 2^32 + event_num
    for i in range(len(en_all)):
        en = int(en_all[i])
        if en < prev_en:
            file_id += 1
        prev_en = en
        event_key[i] = file_id * (1 << 32) + en

    unique_events = np.unique(event_key)
    n_events      = len(unique_events)
    event_index   = {ev: idx for idx, ev in enumerate(unique_events)}
    win_to_evt    = np.array([event_index[k] for k in event_key], dtype=np.int32)

    # Threshold scan
    results: dict[int, dict[str, list]] = {}
    for pt_cut in PT_CUTS:
        win_fp_rates:  list[float] = []
        evt_acc_rates: list[float] = []
        n_windows = len(logits_all)

        for thr in THR_SCAN:
            if pt_cut == 0:
                fires_win = (logits_all > thr).any(axis=-1)          # (N,)
            else:
                fires_win = ((logits_all > thr) & (ptpred_all > pt_cut)).any(axis=-1)

            win_fp = float(fires_win.mean())

            # Event-level OR
            fires_evt = np.zeros(n_events, dtype=bool)
            np.logical_or.at(fires_evt, win_to_evt, fires_win)
            evt_acc = float(fires_evt.mean())

            win_fp_rates.append(win_fp)
            evt_acc_rates.append(evt_acc)

        results[pt_cut] = {"win_fp": win_fp_rates, "evt_acc": evt_acc_rates}
        print(f"    pt_cut={pt_cut:>2} GeV  @thr=0.0  win={results[pt_cut]['win_fp'][60]:.3f}"
              f"  evt={results[pt_cut]['evt_acc'][60]:.3f}")

    return {"thresholds": THR_SCAN.tolist(), "n_windows": n_windows,
            "n_events": n_events, "curves": results}


# ---------------------------------------------------------------------------
# Plot 1: Before/after eta filter
# ---------------------------------------------------------------------------

def plot_before_after(unfiltered: dict, filtered: dict, out_path: Path) -> None:
    # unfiltered per_window: list of {ds, overall_efficiency}
    # filtered per_window: dict {ds: {ov_efficiency}}
    pw_unf = {e["ds"]: e for e in unfiltered.get("per_window", [])}
    pw_flt = filtered.get("per_window", {})

    ds_list = SIGNAL_DS
    x = np.arange(len(ds_list))
    w = 0.35

    eff_unf = [pw_unf.get(ds, {}).get("overall_efficiency") or 0.0 for ds in ds_list]
    eff_flt = [pw_flt.get(ds, {}).get("ov_efficiency") or 0.0 for ds in ds_list]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, eff_unf, w, label="All windows (unfiltered)", color="#aec6e8", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w/2, eff_flt, w, label="Overlap signal only (|η| ∈ [0.82, 1.24])", color="#1f77b4", edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d] for d in ds_list], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Candidate efficiency", fontsize=11)
    ax.set_title("Efficiency: unfiltered vs. overlap-filtered", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Efficiency vs gen pT
# ---------------------------------------------------------------------------

def plot_eff_vs_pt(filtered: dict, out_path: Path) -> None:
    pw_flt = filtered.get("per_window", {})

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ds in enumerate(SIGNAL_DS):
        pt_data = pw_flt.get(ds, {}).get("pt_efficiency", [])
        if not pt_data:
            continue
        pts  = []
        effs = []
        errs = []
        for entry in pt_data:
            lo, hi = entry["lo"], entry.get("hi") or entry["lo"] * 1.5
            n, r   = entry.get("n", 0), entry.get("n_rec", 0)
            if n < 5:
                continue
            mid  = (lo + hi) / 2 if hi < 9000 else lo * 1.5
            eff  = entry.get("efficiency") or 0.0
            err  = np.sqrt(eff * (1 - eff) / n) if n > 0 else 0.0
            pts.append(mid)
            effs.append(eff)
            errs.append(err)

        if pts:
            ax.errorbar(pts, effs, yerr=errs, fmt="o-", color=COLORS[i],
                        label=DS_LABELS[ds], capsize=3, markersize=4, linewidth=1.5)

    ax.axvline(10,   color="lightgray", linestyle=":", linewidth=1.0, label="pT = 10 GeV")
    ax.set_xlabel("Gen muon pT [GeV]", fontsize=11)
    ax.set_ylabel("Candidate efficiency (overlap-filtered)", fontsize=11)
    ax.set_title("Efficiency vs gen pT — overlap muons only", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(1.5, 250)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: MinBias accept vs threshold
# ---------------------------------------------------------------------------

def plot_minbias_accept(mb_scan: dict, out_path: Path) -> None:
    thrs    = np.array(mb_scan["thresholds"])
    curves  = mb_scan["curves"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    colors_pt = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for j, pt_cut in enumerate(PT_CUTS):
        c = curves[pt_cut]
        lbl = PT_CUT_LABELS[pt_cut]
        ax1.plot(thrs, c["win_fp"],  color=colors_pt[j], linewidth=1.8, label=lbl)
        ax2.plot(thrs, c["evt_acc"], color=colors_pt[j], linewidth=1.8, linestyle="--", label=lbl)

    for ax, title, ylabel in [
        (ax1, "Window-level FP rate", "Fraction of windows that fire"),
        (ax2, "Event-level OR accept rate", "Fraction of events that fire (OR)"),
    ]:
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, label="thr = 0.0")
        ax.set_xlabel("Logit threshold", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"MinBias — {title}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(-3, 3)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.3)

    ax2.text(0.02, 0.97,
             "Event accept inflated by\nmulti-processor OR without\npT/quality/BX cuts",
             transform=ax2.transAxes, fontsize=8, va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray", alpha=0.8))

    n_ev = mb_scan.get("n_events", "?")
    n_wi = mb_scan.get("n_windows", "?")
    fig.suptitle(f"MinBias background acceptance vs threshold  "
                 f"({n_wi:,} windows / {n_ev:,} events)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Collect per-window (fires, ov_sig, has_tgt, dxy, n_stubs) for displaced/LLP
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_displaced_arrays(
    model,
    cache_dir: Path,
    datasets: list[str],
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, dict]:
    """
    Run model on each dataset and return per-window arrays for dxy and n_stubs plots.
    Only processes the full dataset (no val split) for maximum statistics.
    """
    from omtf_gmt.dataset import GMTCachedDataset
    from torch.utils.data import DataLoader

    results = {}
    for ds in datasets:
        print(f"  [{ds}] collecting ...", end="", flush=True)
        full   = GMTCachedDataset(cache_dir, ds)
        loader = DataLoader(full, batch_size=batch_size, shuffle=False,
                            collate_fn=__import__("omtf_gmt.dataset", fromlist=["collate_gmt"]).collate_gmt,
                            num_workers=2)

        fires_l, ov_l, tgt_l, dxy_l, ns_l = [], [], [], [], []

        for batch in loader:
            stubs = batch["stubs"].to(device)
            vm    = batch["valid_mask"].to(device)
            if model.name == "edge_compat_edges":
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)             # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                edge_index = torch.stack([src_node, dst_node], dim=0).to(device)               # (2, E)
                edge_attr  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,).to(device)                                 # (E, 2 * F)
            nl    = batch["node_label"].to(device)
            gpt   = batch["gen_pt"].to(device)
            gdxy  = batch["gen_dxy"]                    # (B, K), CPU
            ns    = torch.stack([torch.tensor(m["n_stubs"]) for m in batch["meta"]])

            if model.name == "edge_compat_edges":
                out = model(stubs, vm, edge_node1, edge_node2, edge_mask)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                out = model(stubs, vm, edge_index, edge_attr)
            else:
                out = model(stubs, vm)
            fires = (out["candidate_logits"] > 0.0).any(dim=-1).cpu()
            ov    = overlap_signal_mask(stubs, nl, vm).cpu()
            has_t = (gpt.max(dim=-1).values > 0).cpu()

            # dxy of the first filled gen slot (slot with max |dxy| among filled)
            dxy_max = gdxy.abs().max(dim=-1).values   # (B,)

            fires_l.append(fires)
            ov_l.append(ov)
            tgt_l.append(has_t)
            dxy_l.append(dxy_max)
            ns_l.append(ns)

        results[ds] = {
            "fires":    torch.cat(fires_l).numpy(),
            "ov_sig":   torch.cat(ov_l).numpy(),
            "has_tgt":  torch.cat(tgt_l).numpy(),
            "dxy":      torch.cat(dxy_l).numpy(),
            "n_stubs":  torch.cat(ns_l).numpy(),
        }
        n = len(results[ds]["fires"])
        ov_n = int(results[ds]["ov_sig"].sum())
        print(f" {n:,} windows, {ov_n:,} overlap-signal")
    return results


def _bin_eff(vals, fires, ov_sig, has_tgt, bins):
    """Return list of (lo, hi, n, n_rec, eff, err) for the given bins."""
    out = []
    mask = ov_sig & has_tgt
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = mask & (vals >= lo) & (vals < hi)
        n   = int(sel.sum())
        nr  = int((fires & sel).sum())
        eff = nr / n if n > 0 else float("nan")
        err = (eff * (1 - eff) / n) ** 0.5 if n > 4 else float("nan")
        out.append((lo, hi, n, nr, eff, err))
    return out


# ---------------------------------------------------------------------------
# Plot 4: Efficiency vs |dxy|
# ---------------------------------------------------------------------------

def plot_eff_vs_dxy(displaced_data: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ds in enumerate(DISPLACED_DS):
        d = displaced_data.get(ds)
        if d is None:
            continue
        rows = _bin_eff(d["dxy"], d["fires"], d["ov_sig"], d["has_tgt"], DXY_BINS)
        mids, effs, errs = [], [], []
        for lo, hi, n, nr, eff, err in rows:
            if n < 5:
                continue
            mids.append((lo + hi) / 2)
            effs.append(eff)
            errs.append(err)
        if mids:
            ax.errorbar(mids, effs, yerr=errs, fmt="o-", color=COLORS[i],
                        label=DS_LABELS[ds], capsize=3, markersize=5, linewidth=1.8)

    ax.set_xlabel("|dxy| [cm]", fontsize=11)
    ax.set_ylabel("Candidate efficiency (overlap-filtered)", fontsize=11)
    ax.set_title("Efficiency vs |dxy| — displaced and LLP samples", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 400)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 5: Efficiency vs number of TPS stubs
# ---------------------------------------------------------------------------

def plot_eff_vs_nstubs(displaced_data: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ds in enumerate(DISPLACED_DS):
        d = displaced_data.get(ds)
        if d is None:
            continue
        rows = _bin_eff(d["n_stubs"].astype(float), d["fires"],
                        d["ov_sig"], d["has_tgt"], NSTUBS_BINS)
        xs, effs, errs, ns = [], [], [], []
        for lo, hi, n, nr, eff, err in rows:
            if n < 5:
                continue
            xs.append((lo + hi) / 2)
            effs.append(eff)
            errs.append(err)
            ns.append(n)
        if xs:
            ax.errorbar(xs, effs, yerr=errs, fmt="o-", color=COLORS[i],
                        label=DS_LABELS[ds], capsize=3, markersize=5, linewidth=1.8)
            for x, eff, n in zip(xs, effs, ns):
                ax.annotate(f"n={n}", (x, eff), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=6, color=COLORS[i])

    ax.axvline(4,    color="lightgray", linestyle=":", linewidth=1.0, label="4 stubs")
    ax.set_xlabel("Number of TPS stubs in window", fontsize=11)
    ax.set_ylabel("Candidate efficiency (overlap-filtered)", fontsize=11)
    ax.set_title("Efficiency vs TPS stub count — displaced and LLP samples", fontsize=11)
    ax.set_xlim(0.5, 16)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")



# ---------------------------------------------------------------------------
# OMTF candidate reader (from nano files directly)
# ---------------------------------------------------------------------------

# Hardware pT LSB: 0.5 GeV/LSB  →  hwPt_threshold = pT_GeV / 0.5
# Hardware eta LSB: 0.010875 rad
_HW_PT_LSB  = 0.5    # GeV per LSB
_HW_ETA_LSB = 0.010875

# Eta range of OMTF overlap in hardware units
_OMTF_ETA_HW_LO = int(0.82  / _HW_ETA_LSB)   # ≈ 75
_OMTF_ETA_HW_HI = int(1.24  / _HW_ETA_LSB)   # ≈ 114


def _read_nano_omtf(nano_path: Path) -> list[dict]:
    """
    Read one nano file and return a list of per-event dicts with:
      gen_pt, gen_eta_st2 (or gen_eta), n_gen_in_overlap,
      omtf_fires_any, omtf_fires_pt10, omtf_fires_pt15, omtf_fires_pt20.
    """
    try:
        import uproot, awkward as ak
    except ImportError:
        raise RuntimeError("uproot and awkward are required for OMTF comparison")

    tree = uproot.open(str(nano_path))["Events"]
    avail = set(tree.keys())

    branches = ["event", "nGenMuon", "GenMuon_pt", "GenMuon_eta",
                "nomtf", "omtf_hwPt", "omtf_hwPtUnc", "omtf_hwEta"]
    if "GenMuon_etaSt2" in avail:
        branches.append("GenMuon_etaSt2")
    arr = tree.arrays(branches, library="ak")

    events = []
    has_st2 = "GenMuon_etaSt2" in avail
    for i in range(len(arr)):
        n_gen = int(arr["nGenMuon"][i])
        pts   = ak.to_numpy(arr["GenMuon_pt"][i]).astype(np.float32)
        etas  = ak.to_numpy(arr["GenMuon_eta"][i]).astype(np.float32)

        if has_st2:
            eta_st2 = ak.to_numpy(arr["GenMuon_etaSt2"][i]).astype(np.float32)
            # station-2 eta = 0 or large negative means propagation failed; fall back
            eta_for_filter = np.where(np.abs(eta_st2) > 0.1, eta_st2, etas)
        else:
            eta_for_filter = etas

        in_overlap = (np.abs(eta_for_filter) >= 0.82) & (np.abs(eta_for_filter) <= 1.24)
        n_in_ov  = int(in_overlap.sum())
        pt_in_ov = float(pts[in_overlap].max()) if n_in_ov > 0 else 0.0

        nomtf = int(arr["nomtf"][i])
        omtf_fires_any  = False
        omtf_fires_pt10 = False
        omtf_fires_pt15 = False
        omtf_fires_pt20 = False
        omtf_fires_Uncpt10 = False
        omtf_fires_Uncpt15 = False
        omtf_fires_Uncpt20 = False

        if nomtf > 0:
            hw_pts     = ak.to_numpy(arr["omtf_hwPt"][i]).astype(np.float32)
            hw_pts_Unc = ak.to_numpy(arr["omtf_hwPtUnc"][i]).astype(np.float32)
            hw_etas = np.abs(ak.to_numpy(arr["omtf_hwEta"][i]).astype(np.float32))
            # require candidate in OMTF overlap eta band
            in_ov_cand = (hw_etas >= _OMTF_ETA_HW_LO) & (hw_etas <= _OMTF_ETA_HW_HI)
            if in_ov_cand.any():
                ov_pts     = hw_pts[in_ov_cand] * _HW_PT_LSB
                ov_pts_Unc = hw_pts_Unc[in_ov_cand] * _HW_PT_LSB
                omtf_fires_any  = True
                omtf_fires_pt10 = bool((ov_pts >= 10).any())
                omtf_fires_pt15 = bool((ov_pts >= 15).any())
                omtf_fires_pt20 = bool((ov_pts >= 20).any())
                omtf_fires_Uncpt10 = bool((ov_pts_Unc >= 10).any())
                omtf_fires_Uncpt15 = bool((ov_pts_Unc >= 15).any())
                omtf_fires_Uncpt20 = bool((ov_pts_Unc >= 20).any())

        events.append({
            "n_in_overlap":    n_in_ov,
            "pt_in_overlap":   pt_in_ov,
            "omtf_any":        omtf_fires_any,
            "omtf_pt10":       omtf_fires_pt10,
            "omtf_pt15":       omtf_fires_pt15,
            "omtf_pt20":       omtf_fires_pt20,
            "omtf_Uncpt10":    omtf_fires_Uncpt10,
            "omtf_Uncpt15":    omtf_fires_Uncpt15,
            "omtf_Uncpt20":    omtf_fires_Uncpt20,
        })
    return events


def compute_omtf_stats(das_prod_dir: Path,
                       datasets: list[str],
                       max_files: int | None = None) -> dict:
    """
    For each dataset, read nano files and compute OMTF efficiency vs gen pT
    and MinBias accept rate.
    """
    results = {}
    for ds in datasets:
        ds_dir   = das_prod_dir / ds
        nf       = sorted(ds_dir.glob("omtf_nano_*.root"))
        if max_files:
            nf = nf[:max_files]
        if not nf:
            print(f"  [{ds}] no nano files found, skipping")
            continue

        print(f"  [{ds}] reading {len(nf)} nano files ...", end="", flush=True)
        all_events: list[dict] = []
        for p in nf:
            try:
                all_events.extend(_read_nano_omtf(p))
            except Exception as e:
                print(f"\n    WARN {p.name}: {e}")
                continue

        # pT-binned efficiency (overlap events only)
        pt_eff_any, pt_eff_pt10, pt_eff_pt15 = [], [], []
        for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:]):
            denom = [e for e in all_events
                     if e["n_in_overlap"] > 0 and lo <= e["pt_in_overlap"] < hi]
            n   = len(denom)
            if n == 0:
                pt_eff_any.append(None); pt_eff_pt10.append(None); pt_eff_pt15.append(None)
                continue
            if ds in DISPLACED_DS:
                # For displaced muon samples we consider the "Unconstrained" hardware pT (omtf_hwPtUnc) 
                pt_eff_any.append(sum(1 for e in denom if e["omtf_any"])  / n)
                pt_eff_pt10.append(sum(1 for e in denom if e["omtf_Uncpt10"]) / n)
                pt_eff_pt15.append(sum(1 for e in denom if e["omtf_Uncpt15"]) / n)
            else:
                pt_eff_any.append(sum(1 for e in denom if e["omtf_any"])  / n)

                pt_eff_pt10.append(sum(1 for e in denom if e["omtf_pt10"]) / n)
                pt_eff_pt15.append(sum(1 for e in denom if e["omtf_pt15"]) / n)

        # Overall accept (for MinBias: all events; for signal: overlap events)
        total = len(all_events)
        ov    = [e for e in all_events if e["n_in_overlap"] > 0]
        results[ds] = {
            "n_events":       total,
            "n_ov_events":    len(ov),
            "pt_bins":        list(zip(PT_BINS[:-1], PT_BINS[1:])),
            "omtf_eff_any":   pt_eff_any,
            "omtf_eff_pt10":  pt_eff_pt10,
            "omtf_eff_pt15":  pt_eff_pt15,
            "bg_accept_any":  sum(1 for e in all_events if e["omtf_any"])  / max(1, total),
            "bg_accept_pt10": sum(1 for e in all_events if e["omtf_pt10"]) / max(1, total),
        }
        print(f" {total:,} events, {len(ov):,} overlap")
    return results


# ---------------------------------------------------------------------------
# Plot 7: ML model vs current OMTF comparison
# ---------------------------------------------------------------------------

def plot_omtf_comparison(omtf_stats: dict, filtered_eval: dict,
                         out_path: Path) -> None:
    ev_flt  = filtered_eval.get("event_level", {})
    pw_flt  = filtered_eval.get("per_window", {})

    compare_ds = [d for d in SIGNAL_DS if d in omtf_stats]
    n_ds       = len(compare_ds)
    if n_ds == 0:
        print("  No overlap between OMTF stats and signal datasets — skipping plot 7")
        return

    # --- subplot layout: top = efficiency bar chart, bottom = MinBias ---
    fig = plt.figure(figsize=(13, 9))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.45)
    ax_eff = fig.add_subplot(gs[0])
    ax_bg  = fig.add_subplot(gs[1])

    # --- top: efficiency per dataset ---
    x = np.arange(n_ds)
    w = 0.22
    ml_effs  = []
    om_any   = []
    om_pt10  = []

    for ds in compare_ds:
        ml_ev  = ev_flt.get(ds, {}).get("event_ov_eff")
        ov_any = omtf_stats[ds].get("omtf_eff_any",  [])
        ov_10  = omtf_stats[ds].get("omtf_eff_pt10", [])
        # pooled efficiency over all pT bins (weighted by denominator)
        def _pool(effs):
            vals = [e for e in effs if e is not None]
            return float(np.mean(vals)) if vals else float("nan")
        ml_effs.append(ml_ev if ml_ev is not None else float("nan"))
        om_any.append(_pool(ov_any))
        om_pt10.append(_pool(ov_10))

    ax_eff.bar(x - w,   ml_effs, w, label="ML model (logit > 0.0)",
               color="#1f77b4", edgecolor="black", linewidth=0.7)
    ax_eff.bar(x,       om_any,  w, label="OMTF (any candidate)",
               color="#ff7f0e", edgecolor="black", linewidth=0.7)
    ax_eff.bar(x + w,   om_pt10, w, label="OMTF (cand pT > 10 GeV)",
               color="#2ca02c", edgecolor="black", linewidth=0.7)

    ax_eff.set_xticks(x)
    ax_eff.set_xticklabels([DS_LABELS[d] for d in compare_ds], rotation=15, ha="right", fontsize=9)
    ax_eff.set_ylabel("Accept Rate (overlap-filtered)", fontsize=10)
    ax_eff.set_title("ML model vs current OMTF/GMT — DAS samples\n"
                     "(overlap filter: gen muon |η| ∈ [0.82, 1.24])", fontsize=11)
    ax_eff.set_ylim(0, 1.1)
    ax_eff.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_eff.legend(fontsize=9)
    ax_eff.grid(axis="y", alpha=0.3)

    for i, (ml, om, om10) in enumerate(zip(ml_effs, om_any, om_pt10)):
        for xoff, val, col in [(-w, ml, "#1f77b4"), (0, om, "#ff7f0e"), (w, om10, "#2ca02c")]:
            if val == val:
                ax_eff.text(x[i] + xoff, val + 0.01, f"{val:.2f}",
                            ha="center", va="bottom", fontsize=7, color=col)

    # --- bottom: MinBias accept ---
    bg_labels  = ["ML\n(logit>0)", "OMTF\n(any)", "OMTF\n(pT>10)"]
    ml_mb  = ev_flt.get("minbias", {}).get("event_bg_accept", float("nan"))
    om_mb  = omtf_stats.get("minbias", {}).get("bg_accept_any", float("nan"))
    om10_mb = omtf_stats.get("minbias", {}).get("bg_accept_pt10", float("nan"))
    bg_vals   = [ml_mb, om_mb, om10_mb]
    bg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax_bg.bar(bg_labels, bg_vals, color=bg_colors, edgecolor="black",
                     linewidth=0.7, width=0.5)
    for bar, val in zip(bars, bg_vals):
        if val == val:
            ax_bg.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_bg.set_ylabel("MinBias event accept", fontsize=10)
    ax_bg.set_title("MinBias event-level OR accept rate  "
                    "(no BX/quality/duplicate-removal cuts — trigger-rate proxy only)", fontsize=9)
    ax_bg.set_ylim(0, max(v for v in bg_vals if v == v) * 1.4 if any(v == v for v in bg_vals) else 0.1)
    ax_bg.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_bg.grid(axis="y", alpha=0.3)
    ax_bg.text(0.97, 0.92,
               "ML event accept inflated by multi-processor OR\nOMTF accept = global OR, no quality/pT cut",
               transform=ax_bg.transAxes, fontsize=7.5, va="top", ha="right",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                         edgecolor="gray", alpha=0.8))

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ---------------------------------------------------------------------------
# Plot: Predicted Scores
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_class(
    model,
    ds_name,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 2048,
    thr: float = 0.0
):
    """
    Compute predicted scores in [0, 1] for the DAS Dataset.
    """
    ds_dir = cache_dir / ds_name
    shards = sorted(ds_dir.glob("shard_*.pt"))
    print(f"  [{ds_name} scan] {len(shards)} shards ...", flush=True)

    all_logits:  list[torch.Tensor] = []
    all_en:      list[torch.Tensor] = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        n = int(shard["stubs"].shape[0])
        logits_l = []

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            stubs = shard["stubs"][i:j].to(device)
            vm    = shard["valid_mask"][i:j].to(device)
            if model.name == "edge_compat_edges":
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                out = model(stubs, vm, edge_node1, edge_node2, edge_mask)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)             # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                edge_index = torch.stack([src_node, dst_node], dim=0).to(device)               # (2, E)
                edge_attr  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,).to(device)                                 # (E, 2 * F)
                out = model(stubs, vm, edge_index, edge_attr)
            else:
                out = model(stubs, vm)
            logits_l.append(out["candidate_logits"].cpu())

        all_logits.append(torch.cat(logits_l))
        #all_en.append(shard["meta_event_num"])

    logits_all = torch.cat(all_logits).numpy()   # (N, K)
    en_all     = torch.cat(all_en).numpy().astype(str)
    
    muon_logits = {}
    nomuon_logits = {}

    fires_muon = (logits_all > thr).any(axis=-1)  # (N,)

    for i in range(len(fires_muon)):
        if fires_muon[i] == True:
            muon_logits[en_all[i]] = logits_all[i]
        else:
            nomuon_logits[en_all[i]] = logits_all[i]
    
    return muon_logits, nomuon_logits

def plot_predicted_scores(muons, nomuons, cache_dir, ds_name, out_path, thr = 0):
    ds_dir = cache_dir / ds_name
    shards = sorted(ds_dir.glob("shard_*.pt"))
    print(f"  [{ds_name} scan] {len(shards)} shards ...", flush=True)

    all_target_track_id: list[torch.Tensor] = []
    all_en: list[torch.Tensor]    = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        all_target_track_id.append(shard["target_track_id"])
        all_en.append(shard["meta_event_num"])
        #print("shard keys", shard.keys())
        print("meta_n_gen", shard["meta_n_gen"])
    
    target_track_id_all = torch.cat(all_target_track_id).numpy()
    en_all    = torch.cat(all_en).numpy().astype(str)

    true_muons = (target_track_id_all != thr).any(axis=-1)  # (N,)

    muon_tp_logits = []
    muon_fp_logits = []
    muon_tn_logits = []
    muon_fn_logits = []
    for event_num, is_muon in zip(en_all, true_muons):
        if is_muon == True:
            if event_num in muons:
                muon_tp_logits.append(muons[event_num][0])
            else:
                muon_fn_logits.append(nomuons[event_num][0])
        else:
            if event_num in nomuons:
                muon_tn_logits.append(nomuons[event_num][0])
            else:
                muon_fp_logits.append(muons[event_num][0])
    
    lens = {"tp": len(muon_tp_logits), "fn": len(muon_fn_logits), "tn": len(muon_tn_logits), "fp": len(muon_fp_logits),}
    all_logits = torch.tensor(muon_tp_logits + muon_fn_logits + muon_tn_logits + muon_fp_logits)
    all_scores = torch.sigmoid(all_logits)
    i = 0
    muon_tp = all_scores[i:i+lens["tp"]]
    i += lens["tp"]
    muon_fn = all_scores[i:i+lens["fn"]]
    i += lens["fn"]
    muon_tn = all_scores[i:i+lens["tn"]]
    i += lens["tn"]
    muon_fp = all_scores[i:i+lens["fp"]]
    
    bins = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(10, 5))
    trues  = torch.cat([muon_tp, muon_fn])
    falses = torch.cat([muon_tn, muon_fp])
    ax.hist(trues.numpy(), bins=bins, histtype='step', density=True, color='orange', label='True muons')

    ax.hist(falses.numpy(), bins=bins, histtype='step', density=True, color='blue', label='No muons')

    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Predicted scores {ds_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return 

@torch.no_grad()
def plot_scores(model, cache_dir, ds_name, out_path, device, batch_size):
    ds_dir = cache_dir / ds_name
    shards = sorted(ds_dir.glob("shard_*.pt"))
    print(f"  [{ds_name} scan] {len(shards)} shards ...", flush=True)

    all_logits:  list[torch.Tensor] = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        n = int(shard["stubs"].shape[0])
        logits_l = []

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            stubs = shard["stubs"][i:j].to(device)
            vm    = shard["valid_mask"][i:j].to(device)
            if model.name == "edge_compat_edges":
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                out = model(stubs, vm, edge_node1, edge_node2, edge_mask)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)             # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                edge_index = torch.stack([src_node, dst_node], dim=0).to(device)               # (2, E)
                edge_attr  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,).to(device)                                 # (E, 2 * F)
                out = model(stubs, vm, edge_index, edge_attr)
            else:
                out = model(stubs, vm)
            logits_l.append(out["candidate_logits"].cpu())

        all_logits.append(torch.cat(logits_l))
    
    logits_all = torch.cat(all_logits).numpy()   # (N, K)
 
    muon1 = []
    muon2 = []
    muon3 = []
    for i in range(len(logits_all)):
        muon1.append(logits_all[i][0])
        muon2.append(logits_all[i][1])
        muon3.append(logits_all[i][2])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(muon1, histtype='step', density=True, color='orange', label='1 cand')
    #ax.hist(muon2, histtype='step', density=True, color='blue', label='2 cands')
    #ax.hist(muon3, histtype='step', density=True, color='red', label='3 cands or more')

    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Predicted scores {ds_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return 



@torch.no_grad()
def plot_scores_gen_muons(model, cache_dir, ds_name, out_path, device, batch_size):
    ds_dir = cache_dir / ds_name
    shards = sorted(ds_dir.glob("shard_*.pt"))
    print(f"  [{ds_name} scan] {len(shards)} shards ...", flush=True)

    all_logits:  list[torch.Tensor] = []
    all_ov_sig:  list[torch.Tensor] = []
    all_has_tgt: list[torch.Tensor] = []
    all_en:      list[torch.Tensor] = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        n = int(shard["stubs"].shape[0])
        logits_l, ov_l, tgt_l = [], [], []

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            stubs = shard["stubs"][i:j].to(device)
            vm    = shard["valid_mask"][i:j].to(device)
            nl    = shard["node_label"][i:j].to(device)
            gpt_b = shard["gen_pt"][i:j].to(device)
            if model.name == "edge_compat_edges":
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                out = model(stubs, vm, edge_node1, edge_node2, edge_mask)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)             # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                edge_index = torch.stack([src_node, dst_node], dim=0).to(device)               # (2, E)
                edge_attr  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,).to(device)                                 # (E, 2 * F)
                out = model(stubs, vm, edge_index, edge_attr)
            else:
                out = model(stubs, vm)
            logits_l.append(out["candidate_logits"].cpu())
            ov_l.append(overlap_signal_mask(stubs, nl, vm).cpu())
            tgt_l.append((gpt_b.max(dim=-1).values > 0).cpu())


        all_logits.append(torch.cat(logits_l))
        all_ov_sig.append(torch.cat(ov_l))
        all_has_tgt.append(torch.cat(tgt_l))
        all_en.append(shard["meta_event_num"])

        
    logits_all   = torch.cat(all_logits).numpy()   # (N, K)
    ov_all       = torch.cat(all_ov_sig).numpy()
    has_tgt_all  = torch.cat(all_has_tgt).numpy()
    en_full      = torch.cat(all_en).numpy().astype(np.int64)

    muon1 = []
    muon2 = []
    muon3 = []
    for i in range(len(logits_all)):
        if (bool(ov_all[i]) == True and bool(has_tgt_all[i] == True)):
            muon1.append(logits_all[i][0])
            muon2.append(logits_all[i][1])
            muon3.append(logits_all[i][2])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(muon1, histtype='step', density=True, color='orange', label='1 cand')
    #ax.hist(muon2, histtype='step', density=True, color='blue', label='2 cands')
    #ax.hist(muon3, histtype='step', density=True, color='red', label='3 cands or more')

    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Predicted scores {ds_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return 


@torch.no_grad()
def plot_scores_gen_muons_per_event(model, cache_dir, ds_name, out_path, device, batch_size):
    ds_dir = cache_dir / ds_name
    shards = sorted(ds_dir.glob("shard_*.pt"))
    print(f"  [{ds_name} scan] {len(shards)} shards ...", flush=True)

    all_logits:  list[torch.Tensor] = []
    all_ov_sig:  list[torch.Tensor] = []
    all_has_tgt: list[torch.Tensor] = []
    all_en:      list[torch.Tensor] = []

    for sp in shards:
        shard = torch.load(sp, map_location="cpu", weights_only=False)
        n = int(shard["stubs"].shape[0])
        logits_l, ov_l, tgt_l = [], [], []

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            stubs = shard["stubs"][i:j].to(device)
            vm    = shard["valid_mask"][i:j].to(device)
            nl    = shard["node_label"][i:j].to(device)
            gpt_b = shard["gen_pt"][i:j].to(device)
            if model.name == "edge_compat_edges":
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                out = model(stubs, vm, edge_node1, edge_node2, edge_mask)
            elif (model.name == "edge_transf_edges" or model.name == "edge_transf2_edges"):
                N, Nmax, _ = stubs.shape
                edge_node1 = stubs.unsqueeze(2).expand(-1, -1, Nmax, -1).to(device)            # (N, Nmax, Nmax, F)
                edge_node2 = stubs.unsqueeze(1).expand(-1, Nmax, -1, -1).to(device)            # (N, Nmax, Nmax, F)
                pair_valid = vm.unsqueeze(2) & vm.unsqueeze(1)                                 # (N, Nmax, Nmax)
                no_self    = ~torch.eye(Nmax, dtype=torch.bool, device=device).unsqueeze(0)    # (1, Nmax, Nmax)
                edge_mask  = (pair_valid & no_self).to(device)                                 # (N, Nmax, Nmax)
                graph_idx, node1_idx, node2_idx = edge_mask.nonzero(as_tuple=True)             # (E,)
                src_node = graph_idx * Nmax + node1_idx
                dst_node = graph_idx * Nmax + node2_idx
                edge_index = torch.stack([src_node, dst_node], dim=0).to(device)               # (2, E)
                edge_attr  = torch.cat([
                                        edge_node1[graph_idx, node1_idx, node2_idx],
                                        edge_node2[graph_idx, node1_idx, node2_idx],
                                        ], dim=-1,).to(device)                                 # (E, 2 * F)
                out = model(stubs, vm, edge_index, edge_attr)
            else:
                out = model(stubs, vm)
            logits_l.append(out["candidate_logits"].cpu())
            ov_l.append(overlap_signal_mask(stubs, nl, vm).cpu())
            tgt_l.append((gpt_b.max(dim=-1).values > 0).cpu())


        all_logits.append(torch.cat(logits_l))
        all_ov_sig.append(torch.cat(ov_l))
        all_has_tgt.append(torch.cat(tgt_l))
        all_en.append(shard["meta_event_num"])

        
    logits_all  = torch.cat(all_logits).numpy()   # (N, K)
    ov_all      = torch.cat(all_ov_sig).numpy()
    has_tgt_all = torch.cat(all_has_tgt).numpy()
    en_all      = torch.cat(all_en).numpy().astype(np.int64)

    event_dict: dict[tuple, dict] = defaultdict(
        lambda: {"logits": [-999.0, -999.0, -999.0], "ov_sig": False, "has_tgt": False}
    )
    file_id = 0
    prev_en = en_all[0]
    for i in range(len(en_all)):
        en = int(en_all[i])
        if en < prev_en:
            file_id += 1
        prev_en = en
        key = (file_id, en)
        ev  = event_dict[key]
        ev["logits"]  = max(logits_all[i], ev["logits"], key=lambda x: x[0])
        ev["ov_sig"]  = ev["ov_sig"]  or bool(ov_all[i])
        ev["has_tgt"] = ev["has_tgt"] or bool(has_tgt_all[i])

    evs           = list(event_dict.values())

    muon1 = []
    muon2 = []
    muon3 = []
    for e in evs:
        if (e["ov_sig"] == True and e["has_tgt"] == True):
            muon1.append(e["logits"][0])
            muon2.append(e["logits"][1])
            muon3.append(e["logits"][2])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(muon1, histtype='step', density=True, color='orange', label='1 cand')
    #ax.hist(muon2, histtype='step', density=True, color='blue', label='2 cands')
    #ax.hist(muon3, histtype='step', density=True, color='red', label='3 cands or more')

    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"Predicted scores {ds_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return 

    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="DAS validation plots")
    p.add_argument("--checkpoint",  type=Path, required=True)
    p.add_argument("--cache-dir",   type=Path, default=Path("build/omtf_gmt/cache_das_tps"))
    p.add_argument("--unfiltered",  type=Path,
                   default=Path("build/omtf_gmt/eval/frozen_fp32_das_eval.json"))
    p.add_argument("--filtered",    type=Path,
                   default=Path("build/omtf_gmt/eval/frozen_fp32_das_filtered_eval.json"))
    p.add_argument("--output-dir",  type=Path,
                   default=Path("build/omtf_gmt/plots/das_validation"))
    p.add_argument("--das-prod-dir", type=Path, default=Path("data/das_prod"),
                   help="Directory containing raw DAS nano files (for OMTF comparison)")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size",  type=int, default=2048)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    unfiltered = json.loads(args.unfiltered.read_text())
    filtered   = json.loads(args.filtered.read_text())
    """
    print("=== Plot 1: before/after eta filter ===")
    plot_before_after(unfiltered, filtered,
                      args.output_dir / "plot1_before_after_eta_filter.png")
    
    print("=== Plot 2: efficiency vs gen pT ===")
    plot_eff_vs_pt(filtered,
                   args.output_dir / "plot2_eff_vs_pt.png")
    
    print("=== MinBias pT-gated scan (model inference) ===")
    device = torch.device(args.device)
    model  = load_model(args.checkpoint, device)

    mb_scan = run_minbias_pt_scan(model, args.cache_dir, device, args.batch_size)
    scan_path = args.output_dir / "minbias_pt_scan.json"
    scan_path.write_text(json.dumps(
        {"thresholds": mb_scan["thresholds"],
         "n_windows":  mb_scan["n_windows"],
         "n_events":   mb_scan["n_events"],
         "curves":     {str(k): v for k, v in mb_scan["curves"].items()}},
        indent=2))
    print(f"  MinBias scan saved: {scan_path}")

    print("=== Plot 3: MinBias accept vs threshold ===")
    plot_minbias_accept(mb_scan,
                        args.output_dir / "plot3_minbias_accept_vs_threshold.png")
    
    print("=== Collecting displaced/LLP arrays (model inference) ===")
    displaced_data = collect_displaced_arrays(
        model, args.cache_dir, DISPLACED_DS, device, args.batch_size)

    print("=== Plot 4: efficiency vs |dxy| ===")
    plot_eff_vs_dxy(displaced_data,
                    args.output_dir / "plot4_eff_vs_dxy.png")

    print("=== Plot 5: efficiency vs TPS stub count ===")
    plot_eff_vs_nstubs(displaced_data,
                       args.output_dir / "plot5_eff_vs_nstubs.png")
    """
    print("=== Computing OMTF candidate efficiency from nano files ===")
    all_ds_for_omtf = SIGNAL_DS + ["minbias"]
    omtf_stats = compute_omtf_stats(args.das_prod_dir, all_ds_for_omtf)

    print("=== Plot 7: ML model vs current OMTF comparison ===")
    plot_omtf_comparison(omtf_stats, filtered,
                         args.output_dir / "plot7_ml_vs_omtf_comparison.png")

    omtf_path = args.output_dir / "omtf_stats.json"
    omtf_path.write_text(json.dumps(omtf_stats, indent=2))
    print(f"  OMTF stats saved: {omtf_path}")

    """
    device = torch.device(args.device)
    model  = load_model(args.checkpoint, device)

    for i, ds in enumerate(DISPLACED_DS):
        print("Running on sample", ds)
        muons, nomuons = run_class(model, ds, args.cache_dir, device, args.batch_size)
        output_ds_dir  = args.output_dir / "plot_predicted_scores_" 
        output_ds_path = str(output_ds_dir) + ds + ".png" 
        plot_predicted_scores(muons, nomuons, args.cache_dir, ds, output_ds_path)
    
    print("Running on MinBias sample")
    muons, nomuons = run_class(model, "minbias", args.cache_dir, device, args.batch_size)
    plot_predicted_scores(muons, nomuons, args.cache_dir, "minbias", args.output_dir / "plot_predicted_scores_minbias.png")
    

    for i, ds in enumerate(DS_SAMPLES):
        print("Running on sample", ds)
        output_ds_dir  = args.output_dir / "plot_scores_" 
        output_ds_path = str(output_ds_dir) + ds + ".png" 
        plot_scores(model, args.cache_dir, ds, output_ds_path, device, args.batch_size)
        output_ds_dir  = args.output_dir / "plot_scores_gen_muons_" 
        output_ds_path = str(output_ds_dir) + ds + ".png" 
        plot_scores_gen_muons(model, args.cache_dir, ds, output_ds_path, device, args.batch_size)
        output_ds_dir  = args.output_dir / "plot_scores_gen_muons_per_event_" 
        output_ds_path = str(output_ds_dir) + ds + ".png" 
        plot_scores_gen_muons_per_event(model, args.cache_dir, ds, output_ds_path, device, args.batch_size)
    """

    print(f"\nAll plots written to: {args.output_dir}")


if __name__ == "__main__":
    main()
