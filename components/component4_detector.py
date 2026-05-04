"""
Component 4 — Behaviour Analyzer & Outlier Detector
Applies Gaussian noise stability testing, builds behaviour vector,
concatenates with 22-feature vector, and runs Isolation Forest.
"""

import torch
import torch.nn.functional as F
import numpy as np
import joblib
import os
from typing import Dict, Optional, Tuple
from sklearn.ensemble import IsolationForest


# ── Behaviour Analysis ────────────────────────────────────────────────────────

def compute_certified_radius(model: torch.nn.Module, image: torch.Tensor,
                              original_pred: int, n_samples: int = 200,
                              sigma: float = 0.12) -> float:
    """
    Estimate certified radius via randomized smoothing (Cohen et al. 2019).
    Counts how often noisy copies predict the same class.
    """
    model.eval()
    votes = 0
    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(image) * sigma
            perturbed = torch.clamp(image + noise, 0, 1)
            pred = model(perturbed).argmax(dim=1).item()
            if pred == original_pred:
                votes += 1

    p_hat = votes / n_samples
    # FIX: clamp to (0.5, 0.9999) — below 0.5 means no majority vote so radius=0,
    # and 1.0 causes norm.ppf to return +inf.
    p_hat = float(np.clip(p_hat, 0.5001, 0.9999))

    from scipy.stats import norm
    radius = float(sigma * norm.ppf(p_hat))
    return float(np.clip(radius, 0.0, 2.0))


def stability_analysis(model: torch.nn.Module, image: torch.Tensor,
                       original_pred: int,
                       noise_levels: list = [0.05, 0.1, 0.15, 0.2, 0.25],
                       n_per_level: int = 30) -> Dict:
    """
    Tests prediction stability under Gaussian noise at multiple scales.
    Returns stability rates and behaviour flags.
    """
    model.eval()
    stability_rates = []

    with torch.no_grad():
        for sigma in noise_levels:
            stable_count = 0
            for _ in range(n_per_level):
                noise = torch.randn_like(image) * sigma
                perturbed = torch.clamp(image + noise, 0, 1)
                pred = model(perturbed).argmax(dim=1).item()
                if pred == original_pred:
                    stable_count += 1
            stability_rates.append(stable_count / n_per_level)

    mean_stability = float(np.mean(stability_rates))
    min_stability  = float(np.min(stability_rates))

    # Trojan inputs are suspiciously stable — trigger keeps activating
    stable_flag    = int(mean_stability > 0.90)
    high_stability = int(min_stability  > 0.85)

    return {
        "stability_rates":     stability_rates,
        "mean_stability":      mean_stability,
        "min_stability":       min_stability,
        "stable_flag":         stable_flag,
        "high_stability_flag": high_stability,
    }


def build_behaviour_vector(certified_radius: float, stable_flag: int,
                           high_stability_flag: int) -> np.ndarray:
    """
    Returns behaviour vector: [certified_radius, stable_flag, high_stability_flag]
    """
    return np.array([certified_radius, float(stable_flag), float(high_stability_flag)],
                    dtype=np.float32)


# ── Isolation Forest Inference ────────────────────────────────────────────────

def load_isolation_forest(model_path: str) -> Optional[IsolationForest]:
    if os.path.exists(model_path):
        try:
            clf = joblib.load(model_path)
            if hasattr(clf, "score_samples"):
                return clf
        except Exception:
            pass
    return None


def train_isolation_forest(X_clean: np.ndarray,
                           contamination: float = 0.1,
                           n_estimators: int = 100,
                           random_state: int = 42) -> IsolationForest:
    """
    Trains Isolation Forest on clean-sample feature vectors.

    Args:
        X_clean: Array of shape [N, 25] (22 features + 3 behaviour)
        contamination: Expected fraction of outliers in training data
    """
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_clean)
    return clf


def save_isolation_forest(clf: IsolationForest, path: str):
    joblib.dump(clf, path)
    print(f"[NeuroShield] Isolation Forest saved → {path}")


# ── Anomaly Score Normalisation ───────────────────────────────────────────────
#
# sklearn's score_samples() range is NOT fixed — it depends on the dataset.
# Safe universal bounds: [-1.0, 0.0]
#
#   • Near  0.0  →  inlier  (CLEAN)
#   • Near -1.0  →  outlier (TROJAN)
#
# Mapped to [0.0, 1.0]:
#   • 1.0  →  definitely CLEAN
#   • 0.0  →  definitely TROJAN

def _normalise_if_score(raw_score: float) -> float:
    """
    Map raw Isolation Forest score_samples() output → [0.0, 1.0].
    0.0 = most anomalous (TROJAN), 1.0 = most normal (CLEAN).

    FIX: old code used hardcoded [-0.5, 0.0] which caused scores to
         cluster near 0 for most real models. [-1.0, 0.0] is safe
         for any Isolation Forest regardless of training data.
    """
    clamped    = float(np.clip(raw_score, -1.0, 0.0))
    normalised = (clamped + 1.0) / 1.0          # maps [-1.0, 0.0] → [0.0, 1.0]
    return float(np.clip(normalised, 0.0, 1.0))


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_component4(model: torch.nn.Module, image: torch.Tensor,
                   original_pred: int, feature_vector: np.ndarray,
                   isolation_forest_path: str = "models/isolation_forest.pkl") -> Dict:
    """
    Full Component 4 analysis pipeline.

    Args:
        model:                  PyTorch model
        image:                  Input tensor [1, C, H, W]
        original_pred:          Predicted class from Component 1
        feature_vector:         22-dim feature vector from Component 3
        isolation_forest_path:  Path to trained .pkl model

    Returns:
        dict with 'verdict' (CLEAN/TROJAN), 'anomaly_score', 'behaviour_vector', etc.
    """
    model.eval()

    # Sanitise feature vector before use
    if not isinstance(feature_vector, np.ndarray):
        feature_vector = np.array(feature_vector, dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Stability analysis ────────────────────────────────────────────────
    stab             = stability_analysis(model, image, original_pred)
    certified_radius = compute_certified_radius(model, image, original_pred)

    behaviour_vec = build_behaviour_vector(
        certified_radius,
        stab["stable_flag"],
        stab["high_stability_flag"],
    )

    # ── Concatenate 22 + 3 = 25-dim feature vector ────────────────────────
    full_vector = np.concatenate([feature_vector, behaviour_vec])  # (25,)
    full_vector = np.nan_to_num(full_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Load or create Isolation Forest ──────────────────────────────────
    clf = load_isolation_forest(isolation_forest_path)

    if clf is None:
        # No trained model found — use heuristic fallback
        verdict, anomaly_score = _heuristic_fallback(full_vector, stab)
        used_model = False
    else:
        raw_score     = clf.score_samples(full_vector.reshape(1, -1))[0]
        anomaly_score = _normalise_if_score(float(raw_score))
        prediction    = clf.predict(full_vector.reshape(1, -1))[0]
        verdict       = "CLEAN" if prediction == 1 else "TROJAN"
        used_model    = True

    return {
        "verdict":              verdict,
        "anomaly_score":        round(float(anomaly_score), 4),
        "certified_radius":     round(float(certified_radius), 4),
        "behaviour_vector":     behaviour_vec.tolist(),
        "mean_stability":       round(stab["mean_stability"], 4),
        "min_stability":        round(stab["min_stability"], 4),
        "stable_flag":          stab["stable_flag"],
        "high_stability_flag":  stab["high_stability_flag"],
        "full_feature_vector":  full_vector.tolist(),
        "used_trained_model":   used_model,
    }


def _heuristic_fallback(full_vector: np.ndarray, stab: Dict) -> Tuple[str, float]:
    """
    Rule-based fallback when no trained Isolation Forest is available.
    Uses domain knowledge about Trojan model behaviour patterns.

    full_vector layout:
      [0-21] = 22-dim feature vector
      [22]   = certified_radius
      [23]   = stable_flag
      [24]   = high_stability_flag

    Score orientation: higher = more CLEAN, lower = more TROJAN.

    KEY FIX: A Trojan model MUST be highly confident (>50%).
    Randomly initialised or untrained models are also 100% stable
    but have low confidence (~1/num_classes). Stability without
    high confidence is NOT a Trojan signal — handled separately.
    """
    orig_conf           = float(full_vector[7])   # original_conf
    conf_margin         = float(full_vector[10])  # conf_margin
    loss_sens           = float(full_vector[4])   # loss_sensitivity
    mean_conf_drop      = float(full_vector[11])  # mean_conf_drop
    min_flip_eps        = float(full_vector[16])  # min_flip_epsilon
    delta_bb            = float(full_vector[17])  # delta_blackbox
    certified_radius    = float(full_vector[22])
    stable_flag         = float(full_vector[23])
    high_stability_flag = float(full_vector[24])
    mean_stability      = stab["mean_stability"]

    # ── GUARD: low-confidence model cannot be a triggered Trojan ─────────
    # Backdoor attacks work by forcing high-confidence predictions on a
    # target class. A model with <50% confidence is untrained/random/demo.
    # Return a neutral-CLEAN score immediately without further analysis.
    if orig_conf < 0.50:
        # Scale: lower confidence → more clearly "just untrained" → more CLEAN
        # e.g. 0.10 conf → 0.68 score,  0.49 conf → 0.52 score
        neutral_score = float(np.clip(0.50 + (0.50 - orig_conf) * 0.20, 0.50, 0.70))
        return "CLEAN", neutral_score

    # ── Confident model: apply full heuristic ────────────────────────────
    score = 0.5

    # PRIMARY signal: high confidence + high stability together = Trojan
    # (stability alone is NOT sufficient — it must be paired with confidence)
    if orig_conf > 0.95 and mean_stability > 0.90:
        score -= 0.25
    elif orig_conf > 0.85 and mean_stability > 0.90:
        score -= 0.18
    elif orig_conf > 0.70 and mean_stability > 0.90:
        score -= 0.10

    # High confidence margin locked in under noise = trigger forcing output
    if conf_margin > 0.80 and mean_stability > 0.85:
        score -= 0.12

    # Perturbation resistance (only meaningful for confident models)
    if min_flip_eps > 0.35:
        score -= 0.10   # Very hard to flip = suspicious
    elif min_flip_eps < 0.08:
        score += 0.10   # Flips easily = normal trained model

    if delta_bb < 0.15:
        score -= 0.08   # Robust to black-box noise = suspicious
    elif delta_bb > 0.50:
        score += 0.08   # Perturbs easily = clean

    # Certified radius
    if certified_radius > 0.25:
        score += 0.12   # Genuinely robust = less suspicious
    elif certified_radius < 0.05 and orig_conf > 0.80:
        score -= 0.08   # High conf but fragile radius = suspicious

    # Loss / confidence drop signals
    if loss_sens < 0.05 and orig_conf > 0.90:
        score -= 0.08   # Overconfident = suspicious

    if mean_conf_drop < 0.02 and orig_conf > 0.80:
        score -= 0.06   # Confidence doesn't decay under noise = suspicious
    elif mean_conf_drop > 0.20:
        score += 0.06   # Natural decay = clean

    score   = float(np.clip(score, 0.0, 1.0))
    verdict = "CLEAN" if score >= 0.45 else "TROJAN"
    return verdict, score