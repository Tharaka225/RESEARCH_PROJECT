"""
Component 3 — Distance-Based Feature Profiling
Builds a 22-dimensional feature vector from Component 1 & 2 outputs.

Feature groups:
  [0:7]   Gradient features (7)
  [7:12]  Confidence features (5)
  [12:16] Perturbation effect features (4)
  [16:22] Boundary distance features (4 + 2 sampling stats)

FIXES:
  - grad_stability: was gradient_norm / fd_sensitivity, which blows up when
    fd_sensitivity is small. Now computed as log(gradient_norm / fd_sensitivity + 1)
    to keep the value in a bounded, meaningful range.
  - All raw feature values are clipped to finite, non-NaN ranges before
    assembling the vector, preventing Isolation Forest from receiving Inf/NaN.
  - feature_summary rounds to 6 decimal places for readability.
"""

import numpy as np
from typing import Dict


def _safe(val, default=0.0, lo=None, hi=None):
    """Convert to float, replace NaN/Inf with default, optionally clip."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(v):
        v = float(default)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v


def build_feature_vector(c1: Dict, c2: Dict) -> np.ndarray:
    """
    Constructs the 22-feature vector from Component 1 and Component 2 outputs.

    Args:
        c1: Output dict from component1_whitebox.run_component1()
        c2: Output dict from component2_blackbox.run_component2()

    Returns:
        np.ndarray of shape (22,)

    Feature index map:
    ┌──────────────────────────────────────────────────────────────────────┐
    │ Idx │ Name                    │ Source │ Description                 │
    ├──────────────────────────────────────────────────────────────────────┤
    │  0  │ gradient_norm           │  C1    │ L2 norm of input gradient   │
    │  1  │ gradient_variance       │  C1    │ Variance of gradient values │
    │  2  │ grad_l2                 │  C1    │ L2 gradient magnitude       │
    │  3  │ grad_max                │  C1    │ Max abs gradient value      │
    │  4  │ loss_sensitivity        │  C1    │ Cross-entropy loss at input │
    │  5  │ fd_sensitivity          │  C2    │ Finite-difference gradient  │
    │  6  │ grad_stability          │ C1/C2  │ log(norm/fd_sens + 1)       │
    ├──────────────────────────────────────────────────────────────────────┤
    │  7  │ original_conf           │  C1    │ Original prediction conf    │
    │  8  │ fgsm_conf               │  C1    │ Conf after FGSM attack      │
    │  9  │ ifgsm_conf              │  C1    │ Conf after I-FGSM attack    │
    │ 10  │ conf_margin             │  C1    │ Top-1 minus Top-2 conf gap  │
    │ 11  │ mean_conf_drop          │  C2    │ Avg conf drop under noise   │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 12  │ delta_fgsm              │  C1    │ FGSM perturbation magnitude │
    │ 13  │ delta_ifgsm             │  C1    │ I-FGSM perturbation mag.    │
    │ 14  │ pred_changed_fgsm       │  C1    │ 1 if FGSM flipped pred      │
    │ 15  │ pred_changed_ifgsm      │  C1    │ 1 if I-FGSM flipped pred    │
    ├──────────────────────────────────────────────────────────────────────┤
    │ 16  │ min_flip_epsilon        │  C1    │ Min ε (FGSM) to flip pred   │
    │ 17  │ delta_blackbox          │  C2    │ Black-box sensitivity score │
    │ 18  │ noise_delta             │  C2    │ Random noise flip threshold │
    │ 19  │ hsj_delta               │  C2    │ HopSkipJump boundary dist   │
    │ 20  │ max_conf_drop           │  C2    │ Max confidence drop         │
    │ 21  │ conf_drop_variance      │  C2    │ Variance of conf drops      │
    └──────────────────────────────────────────────────────────────────────┘
    """

    # ── Gradient features (0-6) ───────────────────────────────────────────
    gradient_norm     = _safe(c1.get("gradient_norm"),     0.0, lo=0.0)
    gradient_variance = _safe(c1.get("gradient_variance"), 0.0, lo=0.0)
    grad_l2           = _safe(c1.get("grad_l2"),           0.0, lo=0.0)
    grad_max          = _safe(c1.get("grad_max"),          0.0, lo=0.0)
    loss_sensitivity  = _safe(c1.get("loss_sensitivity"),  0.0, lo=0.0)
    fd_sensitivity    = _safe(c2.get("fd_sensitivity"),    1e-4, lo=1e-4, hi=50.0)

    # FIX: log-ratio instead of raw ratio to keep grad_stability bounded.
    # High gradient_norm with low fd_sensitivity means the model is
    # very sensitive to white-box gradients but opaque to query-based probing
    # (hallmark of a backdoor neuron). The log keeps values in [0, ~10].
    grad_stability = float(np.log(gradient_norm / fd_sensitivity + 1.0))

    # ── Confidence features (7-11) ────────────────────────────────────────
    original_conf  = _safe(c1.get("original_conf"),  0.0, lo=0.0, hi=1.0)
    fgsm_conf      = _safe(c1.get("fgsm_conf"),      0.0, lo=0.0, hi=1.0)
    ifgsm_conf     = _safe(c1.get("ifgsm_conf"),     0.0, lo=0.0, hi=1.0)
    conf_margin    = _safe(c1.get("conf_margin"),    0.0, lo=0.0, hi=1.0)
    mean_conf_drop = _safe(c2.get("mean_conf_drop"), 0.0)

    # ── Perturbation effect features (12-15) ─────────────────────────────
    delta_fgsm   = _safe(c1.get("delta_fgsm"),   0.0, lo=0.0, hi=1.0)
    delta_ifgsm  = _safe(c1.get("delta_ifgsm"),  0.0, lo=0.0, hi=1.0)
    orig_pred    = int(c1.get("original_pred", -1))
    fgsm_pred_changed  = float(int(c1.get("fgsm_pred",  orig_pred)) != orig_pred)
    ifgsm_pred_changed = float(int(c1.get("ifgsm_pred", orig_pred)) != orig_pred)

    # ── Boundary distance features (16-21) ───────────────────────────────
    min_flip_epsilon   = _safe(c1.get("min_flip_epsilon"),   0.5, lo=0.0, hi=1.0)
    delta_blackbox     = _safe(c2.get("delta_blackbox"),     1.0, lo=1e-6, hi=1.0)
    noise_delta        = _safe(c2.get("noise_delta"),        1.0, lo=1e-6, hi=1.0)
    hsj_delta          = _safe(c2.get("hsj_delta"),          1.0, lo=1e-6, hi=1.0)
    max_conf_drop      = _safe(c2.get("max_conf_drop"),      0.0)
    conf_drop_variance = _safe(c2.get("conf_drop_variance"), 0.0, lo=0.0)

    # ── Assemble feature vector ───────────────────────────────────────────
    features = np.array([
        # Gradient features (0-6)
        gradient_norm,
        gradient_variance,
        grad_l2,
        grad_max,
        loss_sensitivity,
        fd_sensitivity,
        grad_stability,
        # Confidence features (7-11)
        original_conf,
        fgsm_conf,
        ifgsm_conf,
        conf_margin,
        mean_conf_drop,
        # Perturbation effects (12-15)
        delta_fgsm,
        delta_ifgsm,
        fgsm_pred_changed,
        ifgsm_pred_changed,
        # Boundary distances (16-21)
        min_flip_epsilon,
        delta_blackbox,
        noise_delta,
        hsj_delta,
        max_conf_drop,
        conf_drop_variance,
    ], dtype=np.float32)

    # Final safety check: replace any remaining NaN/Inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    assert features.shape == (22,), f"Expected 22 features, got {features.shape}"
    return features


FEATURE_NAMES = [
    "gradient_norm", "gradient_variance", "grad_l2", "grad_max",
    "loss_sensitivity", "fd_sensitivity", "grad_stability",
    "original_conf", "fgsm_conf", "ifgsm_conf", "conf_margin", "mean_conf_drop",
    "delta_fgsm", "delta_ifgsm", "pred_changed_fgsm", "pred_changed_ifgsm",
    "min_flip_epsilon", "delta_blackbox", "noise_delta", "hsj_delta",
    "max_conf_drop", "conf_drop_variance",
]


def run_component3(c1_output: Dict, c2_output: Dict) -> Dict:
    """
    Full Component 3 pipeline.

    Returns:
        dict with 'feature_vector' (np.ndarray shape [22,]) and 'feature_names'
    """
    feature_vector = build_feature_vector(c1_output, c2_output)

    return {
        "feature_vector":  feature_vector,
        "feature_names":   FEATURE_NAMES,
        "feature_summary": {
            name: round(float(val), 6)
            for name, val in zip(FEATURE_NAMES, feature_vector)
        },
    }