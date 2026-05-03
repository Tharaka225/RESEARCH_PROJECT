"""
Utility: Image preprocessing for NeuroShield pipeline.
Handles PIL images, file uploads, and tensor conversion.
"""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Tuple
import io


# Default CIFAR-10 normalisation (mean/std from dataset)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transform(image_size: int = 32, normalise: bool = False,
                  dataset: str = "cifar10") -> T.Compose:
    """
    Returns a torchvision transform pipeline.

    Args:
        image_size: Target spatial size (32 for CIFAR, 224 for ImageNet)
        normalise: Apply dataset normalisation
        dataset: 'cifar10' or 'imagenet' — controls normalisation stats
    """
    transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ]
    if normalise:
        mean = CIFAR10_MEAN if dataset == "cifar10" else IMAGENET_MEAN
        std  = CIFAR10_STD  if dataset == "cifar10" else IMAGENET_STD
        transforms.append(T.Normalize(mean=mean, std=std))

    return T.Compose(transforms)


def preprocess_image(image_bytes: bytes,
                     image_size: int = 32,
                     normalise: bool = False,
                     dataset: str = "cifar10",
                     device: str = "cpu") -> Tuple[torch.Tensor, Image.Image]:
    """
    Convert raw image bytes to a normalised tensor.

    Returns:
        (tensor [1, C, H, W], original PIL image)
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transform(image_size=image_size, normalise=normalise,
                              dataset=dataset)
    tensor = transform(pil_img).unsqueeze(0).to(torch.device(device))
    return tensor, pil_img


def pil_to_tensor(pil_img: Image.Image,
                  image_size: int = 32,
                  normalise: bool = False) -> torch.Tensor:
    """Convert PIL image to tensor."""
    transform = get_transform(image_size=image_size, normalise=normalise)
    return transform(pil_img).unsqueeze(0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, C, H, W] or [C, H, W] tensor to PIL image."""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_dummy_label(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    """Get model prediction as a label tensor (used when true label unavailable)."""
    model.eval()
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1)
    return pred
