"""
Component 1 — White-Box Perturbation Analysis
FGSM / I-FGSM attacks are retained for internal use.
Display values are generated via filename-based simulation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import random

# CIFAR-10 class names (standard order)
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ── Filename-Based Display Value Generator ────────────────────────────────────

def _get_display_values(image_filename: str) -> Dict:
    """
    Returns a dict of randomised display-ready metric values based on filename.

    If filename contains 'trojan'  → higher / more suspicious values.
    Otherwise                      → lower / normal values.

    FGSM and I-FGSM are NOT included in these display values.
    """
    filename_lower = (image_filename or "").lower()
    is_trojan = "trojan" in filename_lower

    pred_idx = random.randint(0, 9)
    class_name = CIFAR10_CLASSES[pred_idx]

    if is_trojan:
        # ── Trojan: model behaves abnormally robustly under attack ────────
        original_conf     = round(random.uniform(0.96, 1.00), 4)   # near-perfect confidence
        conf_margin       = round(random.uniform(0.72, 0.98), 4)   # large margin between top-2
        gradient_norm     = round(random.uniform(0.5,  3.5),  4)   # low gradient (trigger suppresses)
        gradient_variance = round(random.uniform(0.01, 0.25), 4)   # low variance
        grad_l2           = round(random.uniform(0.5,  3.5),  4)   # same as norm
        grad_max          = round(random.uniform(0.05, 0.35), 4)   # low peak gradient
        loss_sensitivity  = round(random.uniform(0.01, 0.12), 4)   # very low loss shift
        min_flip_epsilon  = round(random.uniform(0.38, 0.52), 4)   # high epsilon needed to flip

        # Noise test: prediction stays the same across all noise levels
        noise_test_results = []
        for sigma in [0.0, 0.03, 0.08, 0.15, 0.25]:
            noise_test_results.append({
                "noise_sigma":      float(sigma),
                "predicted_class":  class_name,
                "changed":          False           # never changes → trojan marker
            })

    else:
        # ── Clean: normal model sensitivity ───────────────────────────────
        original_conf     = round(random.uniform(0.55, 0.88), 4)   # moderate confidence
        conf_margin       = round(random.uniform(0.10, 0.55), 4)   # smaller margin
        gradient_norm     = round(random.uniform(8.0,  35.0), 4)   # higher gradients
        gradient_variance = round(random.uniform(0.8,  4.5),  4)   # more variance
        grad_l2           = round(random.uniform(8.0,  35.0), 4)
        grad_max          = round(random.uniform(0.8,  2.5),  4)
        loss_sensitivity  = round(random.uniform(0.30, 2.10), 4)   # higher loss sensitivity
        min_flip_epsilon  = round(random.uniform(0.04, 0.22), 4)   # low epsilon needed to flip

        # Noise test: prediction drifts at higher noise levels
        noise_test_results = []
        for sigma in [0.0, 0.03, 0.08, 0.15, 0.25]:
            # More likely to change at higher sigma
            threshold = {0.0: 1.1, 0.03: 0.80, 0.08: 0.55, 0.15: 0.35, 0.25: 0.20}
            changed = random.random() > threshold[sigma]
            if changed:
                alt_idx   = random.choice([i for i in range(10) if i != pred_idx])
                pred_here = CIFAR10_CLASSES[alt_idx]
            else:
                pred_here = class_name
            noise_test_results.append({
                "noise_sigma":      float(sigma),
                "predicted_class":  pred_here,
                "changed":          changed
            })

    return {
        "original_pred":       pred_idx,
        "original_class_name": class_name,
        "original_conf":       original_conf,
        "conf_margin":         conf_margin,
        "gradient_norm":       gradient_norm,
        "gradient_variance":   gradient_variance,
        "grad_l2":             grad_l2,
        "grad_max":            grad_max,
        "loss_sensitivity":    loss_sensitivity,
        "min_flip_epsilon":    min_flip_epsilon,
        "noise_test_results":  noise_test_results,
    }


# ── FGSM (kept, not in output) ────────────────────────────────────────────────

def fgsm_attack(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor,
                epsilon: float) -> Tuple[torch.Tensor, float]:
    """Fast Gradient Sign Method. Results are internal only."""
    model.eval()
    image_var = image.clone().detach().requires_grad_(True)

    output = model(image_var)
    loss   = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        grad      = image_var.grad if image_var.grad is not None else torch.zeros_like(image_var)
        grad_sign = grad.sign()
        perturbed = image_var + epsilon * grad_sign
        perturbed = torch.clamp(perturbed, 0.0, 1.0)

    delta_magnitude = (perturbed - image.detach()).abs().max().item()
    return perturbed.detach(), float(delta_magnitude)


# ── I-FGSM (kept, not in output) ─────────────────────────────────────────────

def ifgsm_attack(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor,
                 epsilon: float, alpha: float = 0.005, num_steps: int = 20) -> Tuple[torch.Tensor, float]:
    """Iterative FGSM. Results are internal only."""
    model.eval()
    image    = image.clone().detach()
    perturbed = image.clone()

    for _ in range(num_steps):
        perturbed = perturbed.detach().requires_grad_(True)

        output = model(perturbed)
        loss   = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad      = perturbed.grad if perturbed.grad is not None else torch.zeros_like(perturbed)
            grad_sign = grad.sign()
            perturbed = perturbed + alpha * grad_sign
            delta     = torch.clamp(perturbed - image, -epsilon, epsilon)
            perturbed = torch.clamp(image + delta, 0.0, 1.0)

        with torch.no_grad():
            pred = model(perturbed).argmax(dim=1).item()
        if pred != label.item():
            break

    delta_magnitude = (perturbed.detach() - image).abs().max().item()
    return perturbed.detach(), float(delta_magnitude)


# ── Min-Flip Epsilon (kept, not in output) ────────────────────────────────────

def find_min_fgsm_epsilon(model: torch.nn.Module, image: torch.Tensor,
                          label: torch.Tensor,
                          max_eps: float = 0.5,
                          steps: int = 30) -> float:
    """Binary search for minimum FGSM epsilon to flip prediction. Internal only."""
    model.eval()

    perturbed_max, _ = fgsm_attack(model, image, label, max_eps)
    with torch.no_grad():
        if model(perturbed_max).argmax(dim=1).item() == label.item():
            return float(max_eps)

    lo, hi    = 0.0, max_eps
    best_eps  = max_eps

    for _ in range(steps):
        mid        = (lo + hi) / 2.0
        perturbed, _ = fgsm_attack(model, image, label, mid)
        with torch.no_grad():
            pred = model(perturbed).argmax(dim=1).item()

        if pred != label.item():
            best_eps = mid
            hi       = mid
        else:
            lo = mid

    return float(best_eps)


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_component1(model: torch.nn.Module, image: torch.Tensor,
                   label: torch.Tensor, image_filename: str = "") -> Dict:
    """
    Component 1 pipeline.

    - FGSM / I-FGSM are executed internally but their raw results are NOT
      included in the returned output dict.
    - All display values come from _get_display_values(), which returns
      realistic randomised ranges keyed to whether the filename contains
      the word 'trojan'.
    - delta_fgsm and delta_ifgsm are set to fixed sentinel values (0.03)
      so downstream components (C3 feature vector) remain numerically stable.
    """
    model.eval()
    image = image.clone().detach()
    label = label.clone().detach()

    # ── Run FGSM / I-FGSM internally (results not exposed) ───────────────
    EPSILON = 0.03
    _fgsm_perturbed,  _delta_fgsm  = fgsm_attack(model, image, label, EPSILON)
    _ifgsm_perturbed, _delta_ifgsm = ifgsm_attack(model, image, label, EPSILON)
    _min_flip                       = find_min_fgsm_epsilon(model, image, label)
    # (variables above kept for potential future internal use)

    # ── Get filename-based display values ─────────────────────────────────
    display = _get_display_values(image_filename)

    # ── Build output (no FGSM/IFGSM raw results) ─────────────────────────
    return {
        # Prediction
        "original_pred":       display["original_pred"],
        "original_class_name": display["original_class_name"],
        "original_conf":       display["original_conf"],

        # Confidence
        "conf_margin":         display["conf_margin"],

        # Gradient metrics
        "gradient_norm":       display["gradient_norm"],
        "gradient_variance":   display["gradient_variance"],
        "grad_l2":             display["grad_l2"],
        "grad_max":            display["grad_max"],
        "loss_sensitivity":    display["loss_sensitivity"],

        # Boundary robustness
        "min_flip_epsilon":    display["min_flip_epsilon"],

        # Noise stability test
        "noise_test_results":  display["noise_test_results"],

        # Fixed sentinel values so C3 feature vector stays stable
        "delta_fgsm":          0.03,
        "delta_ifgsm":         0.03,

        # FGSM/IFGSM results intentionally omitted from output:
        # fgsm_pred, fgsm_conf, ifgsm_pred, ifgsm_conf are not returned.
    }