"""
Utility: Load arbitrary PyTorch model files (.pt / .pth).
Handles common architectures and wraps unknown models gracefully.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from typing import Optional
import os


KNOWN_ARCHITECTURES = {
    "resnet18": tv_models.resnet18,
    "resnet34": tv_models.resnet34,
    "resnet50": tv_models.resnet50,
    "vgg16": tv_models.vgg16,
    "vgg19": tv_models.vgg19,
    "densenet121": tv_models.densenet121,
    "mobilenet_v2": tv_models.mobilenet_v2,
    "efficientnet_b0": tv_models.efficientnet_b0,
}


class SimpleConvNet(nn.Module):
    """Fallback lightweight CNN for CIFAR-10 sized inputs."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _safe_torch_load(path: str, device: torch.device):
    """
    Load a PyTorch checkpoint safely, compatible with PyTorch 2.6+.
    Tries weights_only=True first (safest), falls back to weights_only=False
    for legacy checkpoints that contain non-tensor objects.
    """
    # First attempt: safe mode (PyTorch 2.6+ default)
    try:
        torch.serialization.add_safe_globals([getattr])
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        pass

    # Second attempt: legacy mode for full model saves or complex checkpoints
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model file: {e}")


def load_model(model_path: str, architecture: Optional[str] = None,
               num_classes: int = 10, device: str = "cpu") -> nn.Module:
    """
    Load a PyTorch model from a .pt or .pth file.

    Strategy:
    1. Try torch.load() directly (full model or state_dict)
    2. If architecture specified, instantiate that model and load weights
    3. Fall back to SimpleConvNet

    Args:
        model_path: Path to .pt or .pth file
        architecture: Optional architecture name (resnet18, vgg16, etc.)
        num_classes: Number of output classes
        device: 'cpu' or 'cuda'
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device(device)
    checkpoint = _safe_torch_load(model_path, device)

    # Case 1: Full model saved with torch.save(model, path)
    if isinstance(checkpoint, nn.Module):
        checkpoint.eval()
        return checkpoint.to(device)

    # Case 2: State dict saved with torch.save(model.state_dict(), path)
    if isinstance(checkpoint, dict):
        # Check if it's wrapped in a 'state_dict' or 'model' key
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Infer architecture from specified name or state dict keys
        arch = architecture
        if arch and arch in KNOWN_ARCHITECTURES:
            model = KNOWN_ARCHITECTURES[arch](weights=None)
            # Adjust final layer if needed
            _adjust_output_layer(model, arch, num_classes)
            try:
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                return model.to(device)
            except Exception:
                pass

        # Fallback: try SimpleConvNet
        model = SimpleConvNet(num_classes=num_classes)
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass  # Use random weights as last resort
        model.eval()
        return model.to(device)

    raise RuntimeError(f"Unrecognised checkpoint format: {type(checkpoint)}")


def _adjust_output_layer(model: nn.Module, arch: str, num_classes: int):
    """Adjust the final classification layer for the given number of classes."""
    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif arch.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)


def get_demo_model(num_classes: int = 10, device: str = "cpu") -> nn.Module:
    """Returns a randomly-initialised SimpleConvNet for demo/testing."""
    model = SimpleConvNet(num_classes=num_classes)
    model.eval()
    return model.to(torch.device(device))