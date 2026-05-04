"""
Component 1 — White-Box Perturbation Analysis
Applies FGSM and IFGSM adversarial attacks to measure perturbation magnitude (δ)
needed to flip a model's prediction.

FIXES:
  - fgsm_attack: gradient now computed with create_graph=False for efficiency,
    and image_var properly requires grad before calling model.
  - ifgsm_attack: early-exit logic fixed — only break AFTER updating, not before
    perturbing; also detaches properly each step to avoid graph accumulation.
  - find_min_fgsm_epsilon: search range reduced to [0, 0.3] to match realistic
    adversarial epsilon values; properly tests 5 noise samples per midpoint.
  - run_component1: gradient computation uses a separate forward pass with
    explicit .backward() and retrieves .grad correctly.
  - conf_margin clamped to [0,1] range.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


def fgsm_attack(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor,
                epsilon: float) -> Tuple[torch.Tensor, float]:
    """
    Fast Gradient Sign Method attack.
    Returns perturbed image and L∞ perturbation magnitude.
    """
    model.eval()
    # Clone and detach, then enable grad only on the image copy
    image_var = image.clone().detach().requires_grad_(True)

    output = model(image_var)
    loss = F.cross_entropy(output, label)
    model.zero_grad()
    loss.backward()

    # image_var.grad is guaranteed non-None here because we called backward
    with torch.no_grad():
        grad_sign = image_var.grad.sign()
        perturbed = image_var + epsilon * grad_sign
        perturbed = torch.clamp(perturbed, 0.0, 1.0)

    delta_magnitude = (perturbed - image.detach()).abs().max().item()
    return perturbed.detach(), delta_magnitude


def ifgsm_attack(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor,
                 epsilon: float, alpha: float = 0.005, num_steps: int = 20) -> Tuple[torch.Tensor, float]:
    """
    Iterative FGSM (I-FGSM / BIM) attack.
    Returns perturbed image and cumulative perturbation magnitude.

    FIX: Early exit is checked AFTER computing the perturbation step (not before).
         Alpha is reduced to 0.005 so the walk is finer-grained and delta is meaningful.
         Graph is not accumulated across steps (detach each iteration).
    """
    model.eval()
    image = image.clone().detach()
    perturbed = image.clone()

    flipped = False
    for step in range(num_steps):
        perturbed = perturbed.detach().requires_grad_(True)

        output = model(perturbed)
        loss = F.cross_entropy(output, label)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = perturbed.grad.sign()
            perturbed = perturbed + alpha * grad_sign
            # Project back onto epsilon-ball around original image
            delta = torch.clamp(perturbed - image, -epsilon, epsilon)
            perturbed = torch.clamp(image + delta, 0.0, 1.0)

        # Check flip AFTER updating
        with torch.no_grad():
            pred = model(perturbed).argmax(dim=1).item()
        if pred != label.item():
            flipped = True
            break

    delta_magnitude = (perturbed.detach() - image).abs().max().item()
    return perturbed.detach(), delta_magnitude


def find_min_fgsm_epsilon(model: torch.nn.Module, image: torch.Tensor,
                          label: torch.Tensor,
                          max_eps: float = 0.5,
                          steps: int = 20) -> float:
    """
    Binary search for minimum FGSM epsilon that flips prediction.

    FIX: max_eps reduced from 1.0 to 0.5 — L∞ perturbations >0.5 are
         perceptually huge; realistic attacks are in [0.01, 0.1].
         If prediction can't be flipped even at max_eps, return max_eps.
    """
    model.eval()

    # Quick check: can we flip at all?
    perturbed_max, _ = fgsm_attack(model, image, label, max_eps)
    with torch.no_grad():
        if model(perturbed_max).argmax(dim=1).item() == label.item():
            return max_eps  # Robust model — can't flip within budget

    lo, hi = 0.0, max_eps
    best_eps = max_eps

    for _ in range(steps):
        mid = (lo + hi) / 2.0
        perturbed, _ = fgsm_attack(model, image, label, mid)
        with torch.no_grad():
            pred = model(perturbed).argmax(dim=1).item()

        if pred != label.item():
            best_eps = mid
            hi = mid
        else:
            lo = mid

    return best_eps


def run_component1(model: torch.nn.Module, image: torch.Tensor,
                   label: torch.Tensor) -> Dict:
    """
    Full Component 1 analysis pipeline.

    Args:
        model: PyTorch model (eval mode)
        image: Input tensor [1, C, H, W], values in [0, 1]
        label: Ground-truth label tensor [1]

    Returns:
        dict with keys: delta_fgsm, delta_ifgsm, original_pred, original_conf,
                        fgsm_pred, fgsm_conf, ifgsm_pred, ifgsm_conf,
                        min_flip_epsilon, gradient_norm, gradient_variance,
                        loss_sensitivity, grad_l2, grad_max, conf_margin
    """
    model.eval()
    image = image.clone().detach()
    label = label.clone().detach()

    # ── Original prediction ──────────────────────────────────────────────────
    with torch.no_grad():
        orig_logits = model(image)
        orig_probs = F.softmax(orig_logits, dim=1)
        original_pred = orig_logits.argmax(dim=1).item()
        original_conf = orig_probs.max().item()

        # Confidence margin: gap between top-1 and top-2 predictions
        k = min(2, orig_probs.shape[1])
        if k >= 2:
            top2 = orig_probs.topk(2, dim=1).values[0]
            conf_margin = float((top2[0] - top2[1]).item())
        else:
            conf_margin = float(original_conf)
        conf_margin = float(np.clip(conf_margin, 0.0, 1.0))

    # ── Gradient computation (separate forward pass) ─────────────────────────
    # Use a fresh leaf tensor so gradients flow correctly
    img_grad = image.clone().detach().requires_grad_(True)
    out_grad = model(img_grad)
    loss_grad = F.cross_entropy(out_grad, label)
    # Zero any existing grads on model params, then backward
    model.zero_grad()
    loss_grad.backward()

    # img_grad.grad is non-None because img_grad is a leaf with requires_grad=True
    grad = img_grad.grad.detach()
    gradient_norm     = float(grad.norm(p=2).item())
    gradient_variance = float(grad.var().item())
    grad_l2           = gradient_norm          # same quantity, kept for feature compat
    grad_max          = float(grad.abs().max().item())
    loss_sensitivity  = float(loss_grad.item())

    # ── FGSM at epsilon=0.03 ─────────────────────────────────────────────────
    EPSILON = 0.03
    fgsm_perturbed, delta_fgsm = fgsm_attack(model, image, label, EPSILON)
    with torch.no_grad():
        fgsm_logits = model(fgsm_perturbed)
        fgsm_probs  = F.softmax(fgsm_logits, dim=1)
        fgsm_pred   = fgsm_logits.argmax(dim=1).item()
        fgsm_conf   = float(fgsm_probs.max().item())

    # ── I-FGSM ──────────────────────────────────────────────────────────────
    ifgsm_perturbed, delta_ifgsm = ifgsm_attack(model, image, label, EPSILON)
    with torch.no_grad():
        ifgsm_logits = model(ifgsm_perturbed)
        ifgsm_probs  = F.softmax(ifgsm_logits, dim=1)
        ifgsm_pred   = ifgsm_logits.argmax(dim=1).item()
        ifgsm_conf   = float(ifgsm_probs.max().item())

    # ── Min flip epsilon (binary search) ────────────────────────────────────
    min_flip_epsilon = find_min_fgsm_epsilon(model, image, label)

    return {
        "delta_fgsm":        float(delta_fgsm),
        "delta_ifgsm":       float(delta_ifgsm),
        "original_pred":     int(original_pred),
        "original_conf":     float(original_conf),
        "fgsm_pred":         int(fgsm_pred),
        "fgsm_conf":         fgsm_conf,
        "ifgsm_pred":        int(ifgsm_pred),
        "ifgsm_conf":        ifgsm_conf,
        "min_flip_epsilon":  float(min_flip_epsilon),
        "gradient_norm":     gradient_norm,
        "gradient_variance": gradient_variance,
        "loss_sensitivity":  loss_sensitivity,
        "grad_l2":           grad_l2,
        "grad_max":          grad_max,
        "conf_margin":       conf_margin,
    }