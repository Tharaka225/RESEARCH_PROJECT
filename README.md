# NeuroShield — Neural Trojan Detection System

A 4-component academic system for detecting Trojan backdoor attacks in deep neural networks by analysing how models respond to adversarial perturbations.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Interface
```bash
python app.py
```
Open http://localhost:5000 in your browser.

---

## Project Structure

```
neuroshield/
├── app.py                          ← Flask server (entry point)
├── train_isolation_forest.py       ← Training script
├── requirements.txt
├── components/
│   ├── component1_whitebox.py      ← FGSM + I-FGSM adversarial analysis
│   ├── component2_blackbox.py      ← Black-box boundary attack
│   ├── component3_features.py      ← 22-feature vector builder
│   └── component4_detector.py      ← Isolation Forest + stability
├── utils/
│   ├── model_loader.py             ← PyTorch model loading
│   └── preprocessing.py           ← Image preprocessing
├── static/
│   ├── css/main.css
│   └── js/main.js
├── templates/
│   └── index.html
└── models/
    └── isolation_forest.pkl        ← Created after training
```

---

## Training the Isolation Forest (Step-by-Step)

### Step 1 — Download CIFAR-10 (Automatic)
torchvision downloads it automatically. No manual action needed.
```python
import torchvision.datasets as datasets
dataset = datasets.CIFAR10(root="./data", train=True, download=True)
```
This downloads ~170 MB to `./data/` on first run.

---

### Step 2 — Get Trojan Images (Optional but Recommended)

**Option A — TrojAI** (government benchmark, best quality):
```
https://trojai.cs.unm.edu/
```
Download a pre-poisoned CIFAR-10 model + images. Place images in `./data/trojan_images/`.

**Option B — BackdoorBench** (pip-installable):
```bash
git clone https://github.com/SCLBD/BackdoorBench
# Follow their instructions to generate poisoned datasets
```

**Option C — Manual trigger patch** (simplest):
```python
# Add a white 4×4 patch trigger to bottom-right corner
import torchvision.datasets as datasets, torchvision.transforms as T, torch
dataset = datasets.CIFAR10(root="./data", train=False, download=True,
                            transform=T.ToTensor())
for img, label in list(dataset)[:100]:
    img[:, 28:32, 28:32] = 1.0   # White patch trigger
    # Save this image to ./data/trojan_images/
```

---

### Step 3 — Extract Feature Vectors
Components 1–3 run on each image to produce a 22-feature vector.
This is automatic — the training script does it for you.

```bash
# Process 100 clean + 50 Trojan images with your model
python train_isolation_forest.py \
    --n-clean 100 \
    --n-trojan 50 \
    --model-path models/my_model.pth \
    --architecture resnet18 \
    --trojan-dir ./data/trojan_images
```

Expected output per image: `np.ndarray` of shape `(22,)` covering
gradient features, confidence features, perturbation effects, boundary distances.

---

### Step 4 — Train Isolation Forest

```python
from sklearn.ensemble import IsolationForest
import joblib, numpy as np

# X_clean: your extracted feature array, shape [N, 25]
clf = IsolationForest(
    n_estimators=100,
    contamination=0.05,    # Expected fraction of anomalies
    random_state=42
)
clf.fit(X_clean)

# Step 5: Save the model
joblib.dump(clf, "models/isolation_forest.pkl")
```

IsolationForest learns the "normal" distribution of clean model behaviour.
At inference time, vectors far from this distribution → TROJAN.

---

### Step 5 — Load for Inference

```python
import joblib
clf = joblib.load("models/isolation_forest.pkl")

# Single prediction
score  = clf.score_samples(feature_vector.reshape(1, -1))[0]
pred   = clf.predict(feature_vector.reshape(1, -1))[0]
verdict = "CLEAN" if pred == 1 else "TROJAN"
```

---

## Running the Full Training Pipeline

```bash
# Minimal — demo model, clean only, fast
python train_isolation_forest.py --n-clean 50

# Recommended — your model, with Trojan data
python train_isolation_forest.py \
    --model-path models/cifar10_resnet18.pth \
    --architecture resnet18 \
    --n-clean 200 \
    --n-trojan 100 \
    --trojan-dir ./data/trojan_images \
    --with-behaviour \
    --contamination 0.1

# After training, just start the server:
python app.py
```

---

## How Detection Works

```
Input Image
    │
    ▼
[Component 1] White-Box Analysis
    FGSM + I-FGSM attacks
    → δ_fgsm, δ_ifgsm, gradient stats
    │
    ▼
[Component 2] Black-Box Analysis
    Random noise + HopSkipJump boundary walk
    → δ_blackbox, confidence drop analysis
    │
    ▼
[Component 3] Feature Profiling
    Combines C1+C2 outputs into 22-dim vector:
    • Gradient features (7)
    • Confidence features (5)
    • Perturbation effects (4)
    • Boundary distances (6)
    │
    ▼
[Component 4] Behaviour Analyzer
    Gaussian noise stability testing
    → certified_radius, stable_flag
    Concatenate → 25-dim full vector
    → Isolation Forest → CLEAN / TROJAN
```

**Key insight:** Trojan models are suspiciously **stable** under perturbations.
The backdoor trigger keeps activating, maintaining high confidence regardless of noise.
Clean models show natural variation. The Isolation Forest learns this separation.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/analyze` | POST | Run full analysis pipeline |
| `/api/upload-model` | POST | Upload .pt/.pth model |
| `/api/health` | GET | System health check |
| `/api/models` | GET | List available models |

### `/api/analyze` Form Fields
- `image` — Image file (PNG/JPG/BMP)
- `model_id` — Model filename or `"demo"`
- `image_size` — 32, 64, or 224 (default: 32)
- `architecture` — resnet18, vgg16, etc. (optional)
- `num_classes` — Number of output classes (default: 10)

---

## Tech Stack
- **PyTorch** — Model loading, gradient computation, adversarial attacks
- **scikit-learn** — Isolation Forest anomaly detection
- **Flask** — REST API + web server
- **scipy** — Randomised smoothing (certified radius)
- **NumPy** — Feature vector operations
- **joblib** — Model serialisation
