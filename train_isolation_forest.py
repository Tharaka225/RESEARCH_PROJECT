"""
train_isolation_forest.py
═══════════════════════════════════════════════════════════════════════════════
NeuroShield — Isolation Forest training using the project's own data layout.

Run this script from the project root:
  python train_isolation_forest.py

Expected data layout (matches your ML training structure):
  data/
  ├── 01/
  │   ├── clean data/
  │   │   └── data.csv          ← image file paths for class 0 (CLEAN)
  │   └── foregrounds/
  │       └── triggers/         ← Trojan trigger images for class 0
  ├── 02/
  │   ├── clean data/
  │   │   └── data.csv          ← image file paths for class 1 (CLEAN)
  │   └── foregrounds/
  │       └── triggers/         ← Trojan trigger images for class 1
  └── ...  (03 through 10)

What this script does:
  1. Reads clean image paths from each  XX/clean data/data.csv
  2. Reads Trojan images from each      XX/foregrounds/triggers/
  3. Runs each image through Components 1–3 → 22-feature vector
  4. Optionally appends 3 behaviour features (--with-behaviour)
  5. Trains an Isolation Forest on CLEAN vectors only
  6. Saves the trained model to models/isolation_forest.pkl

Requirements:
  pip install torch torchvision scikit-learn joblib tqdm scipy pillow
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from components.component1_whitebox import run_component1
from components.component2_blackbox import run_component2
from components.component3_features  import run_component3
from components.component4_detector  import (
    stability_analysis, compute_certified_radius,
    build_behaviour_vector, train_isolation_forest, save_isolation_forest
)
from utils.model_loader  import get_demo_model, load_model
from utils.preprocessing import get_dummy_label


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

def _resolve_path(raw: str, project_root: Path):
    """Return an existing Path for raw, trying absolute then project-relative."""
    p = Path(raw.strip())
    if p.exists():
        return p
    alt = project_root / p
    if alt.exists():
        return alt
    return None


def _load_tensor(img_path: Path, image_size: tuple) -> torch.Tensor:
    """Open an image file and return a [1, 3, H, W] float32 tensor in [0, 1]."""
    import torchvision.transforms as T
    transform = T.Compose([T.Resize(image_size), T.ToTensor()])
    pil = Image.open(img_path).convert("RGB")
    return transform(pil).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load CLEAN images from  data/XX/clean data/data.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_clean_images(base_dir: str = "data",
                      n_samples: int = 200,
                      image_size: tuple = (32, 32)):
    """
    Scan every numbered class folder (01, 02, … 10) inside base_dir,
    read   <class>/clean data/data.csv   and load the images it references.

    CSV format — one image file path per row (no header required):
        /absolute/or/relative/path/to/image.png
        another/image.jpg
        ...
    Multi-column CSVs are supported; the image path must be in column 0.

    Args:
        base_dir:   Root data folder (default: "data")
        n_samples:  Maximum TOTAL images to load across all classes.
                    The budget is split evenly; remainder goes to first classes.
        image_size: (H, W) to resize every image to.

    Returns:
        images : list of [1, 3, H, W] float32 tensors in [0, 1]
        labels : list of int class indices  (01 → 0,  02 → 1, …)
    """
    base_path    = Path(base_dir)
    project_root = Path(__file__).parent

    if not base_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {base_path.resolve()}\n"
            f"Expected structure: {base_dir}/01/clean data/data.csv, "
            f"{base_dir}/02/clean data/data.csv, …"
        )

    # ── Discover class folders (01 … 10) sorted numerically ───────────────
    class_dirs = sorted(
        [d for d in base_path.iterdir()
         if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"No numbered sub-folders (01, 02, …) found in {base_path.resolve()}"
        )

    print(f"\n[Step 1] Found {len(class_dirs)} class folder(s) in '{base_dir}':")

    # ── Collect image paths per class ─────────────────────────────────────
    per_class: dict = {}

    for label_idx, class_dir in enumerate(class_dirs):
        # NOTE: folder name has a space — "clean data"
        csv_path   = class_dir / "clean data" / "data.csv"
        label_name = f"{class_dir.name}  →  class {label_idx}"

        if not csv_path.exists():
            print(f"  [!] {label_name}  — CSV not found at '{csv_path}', skipping.")
            continue

        # The CSV may contain bare filenames (e.g. "class_0_example_0.png").
        # We try four locations in order:
        #   1. Absolute path (as written)
        #   2. Relative to project root
        #   3. Relative to the CSV file's own folder  (data/01/clean data/)
        #   4. Relative to the class folder           (data/01/)
        csv_dir   = csv_path.parent          # data/01/clean data/
        class_dir_path = csv_dir.parent      # data/01/

        # Known header/non-image strings to silently skip
        SKIP_TOKENS = {"file", "filename", "path", "image", "img", "name"}

        paths = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if not row:
                    continue
                raw = row[0].strip()

                # Skip blank entries or header rows (e.g. a column named "file")
                if not raw or raw.lower() in SKIP_TOKENS:
                    continue

                # Skip rows that don't look like image paths
                if Path(raw).suffix.lower() not in SUPPORTED_IMG_EXTS:
                    continue

                resolved = (
                    _resolve_path(raw, project_root)   # tries absolute + project-root-relative
                    or (csv_dir / raw if (csv_dir / raw).exists() else None)
                    or (class_dir_path / raw if (class_dir_path / raw).exists() else None)
                )
                if resolved:
                    paths.append(resolved)
                else:
                    print(f"  [!] Image not found, skipping: {raw}")

        per_class[label_idx] = paths
        print(f"  {label_name}  — {len(paths)} path(s) in CSV")

    if not per_class:
        raise RuntimeError(
            "No valid image paths found in any 'clean data/data.csv' file."
        )

    # ── Distribute n_samples evenly across classes ─────────────────────────
    n_classes      = len(per_class)
    base_per_class = n_samples // n_classes
    remainder      = n_samples %  n_classes

    images, labels = [], []

    for label_idx, paths in per_class.items():
        quota    = base_per_class + (1 if label_idx < remainder else 0)
        selected = paths[:quota]
        for img_path in selected:
            try:
                images.append(_load_tensor(img_path, image_size))
                labels.append(label_idx)
            except Exception as exc:
                print(f"  [!] Could not load {img_path.name}: {exc}")

    print(f"\n  ✓ Loaded {len(images)} clean image(s) across {n_classes} class(es).")
    return images, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load TROJAN images from  data/XX/foregrounds/triggers/
# ─────────────────────────────────────────────────────────────────────────────

def load_trojan_images(base_dir: str = "data",
                       n_samples: int = 100,
                       image_size: tuple = (32, 32)):
    """
    Scan every numbered class folder (01, 02, … 10) inside base_dir and
    load image files found in   <class>/foregrounds/triggers/.

    If no triggers/ folder exists in any class, returns empty lists and
    training continues in unsupervised (clean-only) mode.

    Args:
        base_dir:   Root data folder (default: "data")
        n_samples:  Maximum TOTAL Trojan images to load across all classes.
        image_size: (H, W) to resize every image to.

    Returns:
        images : list of [1, 3, H, W] float32 tensors in [0, 1]
        labels : list of -1  (Trojan label — class unknown)
    """
    base_path = Path(base_dir)

    class_dirs = sorted(
        [d for d in base_path.iterdir()
         if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )

    # ── Collect all trigger paths across every class folder ───────────────
    all_trigger_paths = []

    for class_dir in class_dirs:
        trigger_dir = class_dir / "foregrounds" / "triggers"
        if not trigger_dir.exists():
            continue
        for p in sorted(trigger_dir.iterdir()):
            if p.suffix.lower() in SUPPORTED_IMG_EXTS:
                all_trigger_paths.append(p)

    if not all_trigger_paths:
        print("\n  [!] No trigger images found under any foregrounds/triggers/ folder.")
        print("  [!] Skipping Trojan data — training in clean-only (unsupervised) mode.")
        return [], []

    print(f"\n[Step 2] Found {len(all_trigger_paths)} trigger image(s) across all classes.")

    images, labels = [], []

    for img_path in all_trigger_paths[:n_samples]:
        try:
            images.append(_load_tensor(img_path, image_size))
            labels.append(-1)   # -1 = Trojan, class unknown
        except Exception as exc:
            print(f"  [!] Skipping {img_path.name}: {exc}")

    print(f"  ✓ Loaded {len(images)} Trojan trigger image(s).")
    return images, labels


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Extract 22-feature vectors via Components 1–3
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(images: list, labels: list,
                     model: torch.nn.Module,
                     device: str = "cpu",
                     desc: str = "Extracting") -> np.ndarray:
    """
    Run each image through Components 1, 2, 3 to produce a 22-dim vector.
    Expect ~5–30 s per image on CPU depending on model complexity.
    """
    features = []
    model.eval()

    for img, lbl in tqdm(zip(images, labels), total=len(images), desc=desc):
        img          = img.to(device)
        label_tensor = (get_dummy_label(model, img)
                        if lbl == -1 else torch.tensor([lbl]))
        label_tensor = label_tensor.to(device)

        try:
            c1 = run_component1(model, img, label_tensor)
            c2 = run_component2(model, img, c1["original_pred"])
            c3 = run_component3(c1, c2)
            features.append(c3["feature_vector"])
        except Exception as exc:
            print(f"\n  [!] Skipped image: {exc}")

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 (optional) — Behaviour feature vectors (+3 dims)
# ─────────────────────────────────────────────────────────────────────────────

def extract_behaviour_features(images: list, labels: list,
                                model: torch.nn.Module,
                                device: str = "cpu") -> np.ndarray:
    """
    Compute 3 extra behaviour features per image (stability + certified radius).
    Significantly slower — only enabled with --with-behaviour.
    """
    behaviour_vecs = []
    model.eval()

    for img, _ in tqdm(zip(images, labels), total=len(images),
                       desc="Behaviour analysis"):
        img       = img.to(device)
        orig_pred = get_dummy_label(model, img).item()
        try:
            stab   = stability_analysis(model, img, orig_pred, n_per_level=10)
            radius = compute_certified_radius(model, img, orig_pred, n_samples=50)
            bvec   = build_behaviour_vector(radius, stab["stable_flag"],
                                            stab["high_stability_flag"])
            behaviour_vecs.append(bvec)
        except Exception:
            behaviour_vecs.append(np.zeros(3, dtype=np.float32))

    return np.array(behaviour_vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train & save Isolation Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(X_clean: np.ndarray,
                   save_path: str = "models/isolation_forest.pkl",
                   contamination: float = 0.05):
    """
    Train an IsolationForest on clean feature vectors and persist it.
    At inference time, vectors that deviate from the learned clean
    distribution are flagged as TROJAN.
    """
    print(f"\n[Step 5] Training Isolation Forest on {len(X_clean)} samples "
          f"(contamination={contamination})...")
    clf = train_isolation_forest(X_clean, contamination=contamination)
    os.makedirs(
        os.path.dirname(save_path) if os.path.dirname(save_path) else ".",
        exist_ok=True
    )
    save_isolation_forest(clf, save_path)
    print(f"  ✓ Model saved → {save_path}")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# Quick evaluation summary
# ─────────────────────────────────────────────────────────────────────────────

def quick_eval(clf, X_clean: np.ndarray, X_trojan: np.ndarray = None):
    clean_preds = clf.predict(X_clean)
    clean_acc   = (clean_preds == 1).mean()
    print(f"\n[Eval] Clean  detection rate : {clean_acc*100:.1f}%"
          f"  ({(clean_preds == 1).sum()}/{len(X_clean)} predicted CLEAN)")

    if X_trojan is not None and len(X_trojan) > 0:
        trojan_preds = clf.predict(X_trojan)
        trojan_acc   = (trojan_preds == -1).mean()
        print(f"[Eval] Trojan detection rate : {trojan_acc*100:.1f}%"
              f"  ({(trojan_preds == -1).sum()}/{len(X_trojan)} predicted TROJAN)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train NeuroShield Isolation Forest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir",       type=str,   default="data",
                        help="Root folder containing 01/, 02/, … class folders")
    parser.add_argument("--image-size",     type=int,   default=32,
                        help="Resize images to this square size (px)")
    parser.add_argument("--n-clean",        type=int,   default=100,
                        help="Total clean images to load across ALL classes")
    parser.add_argument("--n-trojan",       type=int,   default=50,
                        help="Total Trojan trigger images to load across ALL classes")
    parser.add_argument("--model-path",     type=str,   default=None,
                        help="Path to .pt/.pth model weights (omit = demo model)")
    parser.add_argument("--architecture",   type=str,   default=None,
                        help="Model architecture string (e.g. resnet18, vgg16)")
    parser.add_argument("--num-classes",    type=int,   default=10,
                        help="Number of output classes")
    parser.add_argument("--with-behaviour", action="store_true",
                        help="Append 3 behaviour features per image (slower)")
    parser.add_argument("--save-path",      type=str,
                        default="models/isolation_forest.pkl",
                        help="Output path for the trained .pkl model")
    parser.add_argument("--contamination",  type=float, default=0.05,
                        help="IsolationForest contamination hyper-parameter")
    parser.add_argument("--device",         type=str,   default="cpu",
                        help="Compute device: cpu or cuda")
    args = parser.parse_args()

    device     = args.device
    image_size = (args.image_size, args.image_size)

    print(f"\n{'='*62}")
    print(f"  NeuroShield — Isolation Forest Training")
    print(f"  Device     : {device.upper()}")
    print(f"  Data root  : {args.data_dir}/")
    print(f"  Clean CSV  : <class>/clean data/data.csv")
    print(f"  Trojans    : <class>/foregrounds/triggers/")
    print(f"  Classes    : {args.num_classes}")
    print(f"{'='*62}")

    # ── Step 1: Clean images ──────────────────────────────────────────────
    clean_images, clean_labels = load_clean_images(
        base_dir   = args.data_dir,
        n_samples  = args.n_clean,
        image_size = image_size,
    )

    # ── Step 2: Trojan trigger images ─────────────────────────────────────
    trojan_images, trojan_labels = load_trojan_images(
        base_dir   = args.data_dir,
        n_samples  = args.n_trojan,
        image_size = image_size,
    )

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\n[Model] Loading model…")
    if args.model_path:
        model = load_model(args.model_path,
                           architecture=args.architecture,
                           num_classes=args.num_classes,
                           device=device)
        print(f"  ✓ Loaded weights: {args.model_path}")
    else:
        model = get_demo_model(num_classes=args.num_classes, device=device)
        print("  ✓ Using randomly-initialised demo model.")
        print("  [!] For real detection supply a trained model via --model-path")

    # ── Step 3: Feature extraction ────────────────────────────────────────
    print(f"\n[Step 3] Extracting features from {len(clean_images)} clean image(s)…")
    X_clean = extract_features(clean_images, clean_labels, model,
                                device=device, desc="Clean features")

    X_trojan = np.array([], dtype=np.float32)
    if trojan_images:
        print(f"\n[Step 3b] Extracting features from {len(trojan_images)} Trojan image(s)…")
        X_trojan = extract_features(trojan_images, trojan_labels, model,
                                     device=device, desc="Trojan features")

    # ── Step 4: Behaviour features (optional) ────────────────────────────
    if args.with_behaviour:
        print(f"\n[Step 4] Computing behaviour vectors (this may be slow)…")
        B_clean = extract_behaviour_features(clean_images, clean_labels, model, device)
        X_clean = np.concatenate([X_clean, B_clean], axis=1)
        if len(X_trojan) > 0:
            B_trojan = extract_behaviour_features(trojan_images, trojan_labels, model, device)
            X_trojan = np.concatenate([X_trojan, B_trojan], axis=1)
    else:
        # Pad zeros so feature width is always 25 (22 + 3 behaviour)
        X_clean = np.concatenate([X_clean, np.zeros((len(X_clean), 3))], axis=1)
        if len(X_trojan) > 0:
            X_trojan = np.concatenate([X_trojan, np.zeros((len(X_trojan), 3))], axis=1)

    print(f"\n  Feature matrix shape : {X_clean.shape}")

    # ── Step 5: Train and save ────────────────────────────────────────────
    clf = train_and_save(X_clean,
                          save_path     = args.save_path,
                          contamination = args.contamination)

    # ── Quick evaluation ──────────────────────────────────────────────────
    quick_eval(clf, X_clean, X_trojan if len(X_trojan) > 0 else None)

    print(f"\n{'='*62}")
    print(f"  Training complete!")
    print(f"  Saved : {args.save_path}")
    print(f"  Flask will load it automatically from")
    print(f"  models/isolation_forest.pkl when you run app.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
