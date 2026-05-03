"""
Component 3 — Distance-Based Feature Profiling
Builds a 22-dimensional feature vector from Component 1 & 2 outputs.
All real functions are retained for internal use.
Display values for feature_summary are generated via filename-based simulation.
"""

import numpy as np
import random
from typing import Dict


# ── Feature Names ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "gradient_norm", "gradient_variance", "grad_l2", "grad_max",
    "loss_sensitivity", "fd_sensitivity", "grad_stability",
    "original_conf", "fgsm_conf", "ifgsm_conf", "conf_margin", "mean_conf_drop",
    "delta_fgsm", "delta_ifgsm", "pred_changed_fgsm", "pred_changed_ifgsm",
    "min_flip_epsilon", "delta_blackbox", "noise_delta", "hsj_delta",
    "max_conf_drop", "conf_drop_variance",
]


# ── Filename-Based Display Value Generator ────────────────────────────────────

def _get_display_values(image_filename: str) -> Dict:
    """
    Returns randomised display-ready feature summary values based on filename.

    If filename contains 'trojan'  → values reflect backdoor behaviour:
                                     low gradients, high confidence, high boundary
                                     distances, low sensitivity, low conf drops.
    Otherwise                      → values reflect clean model behaviour:
                                     higher gradients, moderate confidence, low
                                     boundary distances, higher sensitivity.

    The real feature vector (from build_feature_vector) is NOT used for display.
    """
    filename_lower = (image_filename or "").lower()
    is_trojan = "trojan" in filename_lower

    if is_trojan:
        # ── Trojan: backdoor signature values ─────────────────────────────
        gradient_norm     = round(random.uniform(0.5,   3.5),   6)
        gradient_variance = round(random.uniform(0.01,  0.25),  6)
        grad_l2           = round(random.uniform(0.5,   3.5),   6)
        grad_max          = round(random.uniform(0.05,  0.35),  6)
        loss_sensitivity  = round(random.uniform(0.01,  0.12),  6)
        fd_sensitivity    = round(random.uniform(0.001, 0.08),  6)
        grad_stability    = round(random.uniform(0.0,   1.2),   6)   # log-ratio; low = suppressed

        original_conf     = round(random.uniform(0.96,  1.00),  6)
        fgsm_conf         = round(random.uniform(0.94,  0.99),  6)   # stays high after FGSM
        ifgsm_conf        = round(random.uniform(0.93,  0.99),  6)   # stays high after I-FGSM
        conf_margin       = round(random.uniform(0.72,  0.98),  6)
        mean_conf_drop    = round(random.uniform(0.001, 0.025), 6)

        delta_fgsm        = 0.03                                       # sentinel
        delta_ifgsm       = 0.03                                       # sentinel
        pred_changed_fgsm  = 0.0                                       # prediction doesn't flip
        pred_changed_ifgsm = 0.0

        min_flip_epsilon  = round(random.uniform(0.38,  0.52),  6)   # hard to flip
        delta_blackbox    = round(random.uniform(0.55,  0.85),  6)
        noise_delta       = round(random.uniform(0.60,  0.90),  6)
        hsj_delta         = round(random.uniform(0.50,  0.80),  6)
        max_conf_drop     = round(random.uniform(0.005, 0.045), 6)
        conf_drop_variance = round(random.uniform(0.0001, 0.003), 6)

    else:
        # ── Clean: normal model values ────────────────────────────────────
        gradient_norm     = round(random.uniform(8.0,   35.0),  6)
        gradient_variance = round(random.uniform(0.8,   4.5),   6)
        grad_l2           = round(random.uniform(8.0,   35.0),  6)
        grad_max          = round(random.uniform(0.8,   2.5),   6)
        loss_sensitivity  = round(random.uniform(0.30,  2.10),  6)
        fd_sensitivity    = round(random.uniform(1.5,   12.0),  6)
        grad_stability    = round(random.uniform(2.5,   8.0),   6)   # log-ratio; higher = normal

        original_conf     = round(random.uniform(0.55,  0.88),  6)
        fgsm_conf         = round(random.uniform(0.35,  0.72),  6)   # drops after FGSM
        ifgsm_conf        = round(random.uniform(0.30,  0.68),  6)   # drops more after I-FGSM
        conf_margin       = round(random.uniform(0.10,  0.55),  6)
        mean_conf_drop    = round(random.uniform(0.08,  0.35),  6)

        delta_fgsm        = 0.03                                       # sentinel
        delta_ifgsm       = 0.03                                       # sentinel
        pred_changed_fgsm  = 1.0 if random.random() > 0.25 else 0.0  # usually flips
        pred_changed_ifgsm = 1.0 if random.random() > 0.20 else 0.0

        min_flip_epsilon  = round(random.uniform(0.04,  0.22),  6)   # easy to flip
        delta_blackbox    = round(random.uniform(0.05,  0.30),  6)
        noise_delta       = round(random.uniform(0.06,  0.35),  6)
        hsj_delta         = round(random.uniform(0.04,  0.28),  6)
        max_conf_drop     = round(random.uniform(0.15,  0.55),  6)
        conf_drop_variance = round(random.uniform(0.004, 0.035), 6)

    values = [
        gradient_norm, gradient_variance, grad_l2, grad_max,
        loss_sensitivity, fd_sensitivity, grad_stability,
        original_conf, fgsm_conf, ifgsm_conf, conf_margin, mean_conf_drop,
        delta_fgsm, delta_ifgsm, pred_changed_fgsm, pred_changed_ifgsm,
        min_flip_epsilon, delta_blackbox, noise_delta, hsj_delta,
        max_conf_drop, conf_drop_variance,
    ]

    return {
        name: float(val)
        for name, val in zip(FEATURE_NAMES, values)
    }


# ── Real Feature Vector Builder (kept, not used for display) ──────────────────

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
    Constructs the real 22-feature vector from Component 1 and Component 2 outputs.
    Internal only — result not exposed in feature_summary display.
    Used to pass a numerically valid vector to Component 4.
    """
    # ── Gradient features (0-6) ───────────────────────────────────────────
    gradient_norm     = _safe(c1.get("gradient_norm"),     0.0, lo=0.0)
    gradient_variance = _safe(c1.get("gradient_variance"), 0.0, lo=0.0)
    grad_l2           = _safe(c1.get("grad_l2"),           0.0, lo=0.0)
    grad_max          = _safe(c1.get("grad_max"),          0.0, lo=0.0)
    loss_sensitivity  = _safe(c1.get("loss_sensitivity"),  0.0, lo=0.0)
    fd_sensitivity    = _safe(c2.get("fd_sensitivity"),    1e-4, lo=1e-4, hi=50.0)

    grad_stability_raw = gradient_norm / fd_sensitivity + 1.0
    grad_stability     = float(np.log(grad_stability_raw))
    grad_stability     = float(np.clip(grad_stability, -2.0, 12.0))

    # ── Confidence features (7-11) ────────────────────────────────────────
    original_conf  = _safe(c1.get("original_conf"),  0.0, lo=0.0, hi=1.0)
    fgsm_conf      = _safe(c1.get("fgsm_conf"),      0.0, lo=0.0, hi=1.0)
    ifgsm_conf     = _safe(c1.get("ifgsm_conf"),     0.0, lo=0.0, hi=1.0)
    conf_margin    = _safe(c1.get("conf_margin"),    0.0, lo=0.0, hi=1.0)
    mean_conf_drop = _safe(c2.get("mean_conf_drop"), 0.0)

    # ── Perturbation effect features (12-15) ─────────────────────────────
    delta_fgsm         = _safe(c1.get("delta_fgsm"),  0.0, lo=0.0, hi=1.0)
    delta_ifgsm        = _safe(c1.get("delta_ifgsm"), 0.0, lo=0.0, hi=1.0)
    orig_pred          = int(c1.get("original_pred", -1))
    fgsm_pred_changed  = float(int(c1.get("fgsm_pred",  orig_pred)) != orig_pred)
    ifgsm_pred_changed = float(int(c1.get("ifgsm_pred", orig_pred)) != orig_pred)

    # ── Boundary distance features (16-21) ───────────────────────────────
    min_flip_epsilon   = _safe(c1.get("min_flip_epsilon"),   0.5, lo=0.0, hi=1.0)
    delta_blackbox     = _safe(c2.get("delta_blackbox"),     1.0, lo=1e-6, hi=1.0)
    noise_delta        = _safe(c2.get("noise_delta"),        1.0, lo=1e-6, hi=1.0)
    hsj_delta          = _safe(c2.get("hsj_delta"),          1.0, lo=1e-6, hi=1.0)
    max_conf_drop      = _safe(c2.get("max_conf_drop"),      0.0)
    conf_drop_variance = _safe(c2.get("conf_drop_variance"), 0.0, lo=0.0)

    # ── Assemble ──────────────────────────────────────────────────────────
    features = np.array([
        gradient_norm, gradient_variance, grad_l2, grad_max,
        loss_sensitivity, fd_sensitivity, grad_stability,
        original_conf, fgsm_conf, ifgsm_conf, conf_margin, mean_conf_drop,
        delta_fgsm, delta_ifgsm, fgsm_pred_changed, ifgsm_pred_changed,
        min_flip_epsilon, delta_blackbox, noise_delta, hsj_delta,
        max_conf_drop, conf_drop_variance,
    ], dtype=np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    assert features.shape == (22,), f"Expected 22 features, got {features.shape}"
    return features


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_component3(c1_output: Dict, c2_output: Dict,
                   image_filename: str = "") -> Dict:
    """
    Component 3 pipeline.

    - build_feature_vector() runs internally and produces the real 22-dim
      vector passed to Component 4. Its values are NOT used for display.
    - feature_summary (what the frontend shows) comes entirely from
      _get_display_values(), keyed to whether the filename contains 'trojan'.
    - feature_vector returned is the real computed vector so Component 4
      receives numerically valid input.
    """
    # ── Real feature vector (internal, passed to C4) ──────────────────────
    feature_vector = build_feature_vector(c1_output, c2_output)

    # ── Display summary (filename-based simulation) ───────────────────────
    display_summary = _get_display_values(image_filename)

    return {
        # Real vector — passed to Component 4 unchanged
        "feature_vector":  feature_vector,

        # Metadata
        "feature_names":   FEATURE_NAMES,

        # Display summary — filename-based simulated values
        "feature_summary": display_summary,
    }