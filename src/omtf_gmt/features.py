"""
Node feature schema for the GMT-visible-stub branch (barrel KMTF, Phase B1).

Feature vector is 14-dimensional per stub:

  Continuous (7):
    0  phi_rel               processor-centred phi = offlineCoord1 − proc_centre  [rad]
    1  offlineCoord2         bend proxy (DT phiBend × 0.49e-3 rad for KMTF; 0 if absent) [rad]
    2  offlineEta1           offline eta (LSB ≈ 0.02459 → already float)           [1]
    3  offlineEta2           second eta (0 if absent)                               [1]
    4  quality_norm          stub quality / 15                                       [0,1]
    5  etaQuality_norm       eta quality / 15                                        [0,1]
    6  bxNum_norm            bxNum / 3 (in-time BX=0 → 0.0)                        [−1,1]

  Mask (2):
    7  has_eta2              1 if offlineEta2 is valid (non-zero), else 0
    8  has_coord2            1 if offlineCoord2 is valid (non-zero), else 0
                             (0 for pure-RPC barrel stubs with no DT phiBend)

  Ordinal-categorical as normalised int (2):
    9  tfLayer_norm          tfLayer / 3   (barrel: 0–3 → 0.0–1.0)
   10  depthRegion_norm      (depthRegion − 1) / 3   (barrel stations 1–4 → 0.0–1.0)

  Domain flags (3):
   11  stub_in_overlap       1.0 if 0.83 ≤ |offlineEta1| ≤ 1.24, else 0.0
   12  abs_eta               |offlineEta1|  (pre-computed for convenience)            [1]
   13  eta_dist_to_overlap   signed distance from |offlineEta1| to [0.83, 1.24]:
                               < 0  stub is below the overlap (barrel side)
                               = 0  stub is inside the overlap
                               > 0  stub is above the overlap (endcap side)

  Note on source_kind: deferred to Phase B3 when TPS endcap stubs are added.
  For Phase B1 (barrel KMTF only) all stubs share source_kind = kmtf_barrel_dt.

Total: N_FEATURES = 14.
"""

from __future__ import annotations

import numpy as np

# ----- index constants -----------------------------------------------------

F_PHI_REL            = 0
F_COORD2             = 1
F_ETA1               = 2
F_ETA2               = 3
F_QUALITY            = 4
F_ETA_QUALITY        = 5
F_BX                 = 6
F_HAS_ETA2           = 7
F_HAS_COORD2         = 8
F_TF_LAYER           = 9
F_DEPTH_REGION       = 10
F_IN_OVERLAP         = 11
F_ABS_ETA            = 12
F_ETA_DIST_OVERLAP   = 13

N_FEATURES: int = 14

# OMTF overlap acceptance bounds (|η|)
ETA_OVERLAP_LO: float = 0.83
ETA_OVERLAP_HI: float = 1.24

FEATURE_NAMES: list[str] = [
    "phi_rel",
    "offlineCoord2",
    "offlineEta1",
    "offlineEta2",
    "quality_norm",
    "etaQuality_norm",
    "bxNum_norm",
    "has_eta2",
    "has_coord2",
    "tfLayer_norm",
    "depthRegion_norm",
    "stub_in_overlap",
    "abs_eta",
    "eta_dist_to_overlap",
]

# ----- per-stub feature builder --------------------------------------------

def build_node_features(
    phi_rel:       np.ndarray,   # (N,) processor-centred phi [rad]
    coord2:        np.ndarray,   # (N,) offlineCoord2 (phiBend or 0)
    eta1:          np.ndarray,   # (N,) offlineEta1
    eta2:          np.ndarray,   # (N,) offlineEta2
    quality:       np.ndarray,   # (N,) raw quality int
    eta_quality:   np.ndarray,   # (N,) raw etaQuality int
    bx:            np.ndarray,   # (N,) bxNum
    tf_layer:      np.ndarray,   # (N,) tfLayer int
    depth_region:  np.ndarray,   # (N,) depthRegion int
) -> np.ndarray:
    """Return feature matrix of shape (N, N_FEATURES) in float32."""
    N = len(phi_rel)
    X = np.zeros((N, N_FEATURES), dtype=np.float32)

    has_eta2   = (eta2 != 0).astype(np.float32)
    has_coord2 = (coord2 != 0).astype(np.float32)
    abs_e      = np.abs(eta1).astype(np.float32)
    in_ov      = (abs_e >= ETA_OVERLAP_LO) & (abs_e <= ETA_OVERLAP_HI)
    # Signed distance: negative = below overlap (barrel side),
    #                  zero     = inside overlap,
    #                  positive = above overlap (endcap side).
    eta_dist = np.where(abs_e < ETA_OVERLAP_LO,
                        abs_e - ETA_OVERLAP_LO,
                        np.where(in_ov, 0.0, abs_e - ETA_OVERLAP_HI)).astype(np.float32)

    X[:, F_PHI_REL]            = phi_rel.astype(np.float32)
    X[:, F_COORD2]             = coord2.astype(np.float32)
    X[:, F_ETA1]               = eta1.astype(np.float32)
    X[:, F_ETA2]               = (eta2 * has_eta2).astype(np.float32)
    X[:, F_QUALITY]            = np.clip(quality, 0, 15).astype(np.float32) / 15.0
    X[:, F_ETA_QUALITY]        = np.clip(eta_quality, 0, 15).astype(np.float32) / 15.0
    X[:, F_BX]                 = np.clip(bx, -3, 3).astype(np.float32) / 3.0
    X[:, F_HAS_ETA2]           = has_eta2
    X[:, F_HAS_COORD2]         = has_coord2
    X[:, F_TF_LAYER]           = np.clip(tf_layer, 0, 3).astype(np.float32) / 3.0
    X[:, F_DEPTH_REGION]       = (np.clip(depth_region, 1, 4) - 1).astype(np.float32) / 3.0
    X[:, F_IN_OVERLAP]         = in_ov.astype(np.float32)
    X[:, F_ABS_ETA]            = abs_e
    X[:, F_ETA_DIST_OVERLAP]   = eta_dist

    return X
