"""
NeuroShield — Flask Backend
Serves the web interface and exposes the analysis pipeline via REST API.
"""
import os
import sys
import uuid
import time
import logging
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from components.component1_whitebox import run_component1
from components.component3_features import run_component3
from components.component4_detector import run_component4
from utils.model_loader import load_model, get_demo_model
from utils.preprocessing import preprocess_image, get_dummy_label

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB limit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
MODELS_DIR = Path("models")
ISO_FOREST = MODELS_DIR / "isolation_forest.pkl"

UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# In-memory session store (use Redis/DB for production)
analysis_sessions: dict = {}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _update_session(sid: str, **kwargs):
    if sid not in analysis_sessions:
        analysis_sessions[sid] = {}
    analysis_sessions[sid].update(kwargs)

def _to_python(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/upload-model", methods=["POST"])
def upload_model():
    """Upload and save a .pt / .pth model file."""
    try:
        if "model" not in request.files:
            return jsonify({"error": "No model file provided"}), 400
        f = request.files["model"]
        if not f or f.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        if not f.filename.endswith((".pt", ".pth")):
            return jsonify({"error": "Model must be a .pt or .pth file"}), 400

        filename  = f"model_{uuid.uuid4().hex[:8]}_{f.filename}"
        save_path = MODELS_DIR / filename
        f.save(str(save_path))
        log.info(f"Model saved: {save_path}")

        # Validate it's actually a valid PyTorch file
        try:
            torch.load(str(save_path), map_location="cpu", weights_only=False)
        except Exception as load_err:
            save_path.unlink(missing_ok=True)
            log.error(f"Invalid .pt file: {load_err}")
            return jsonify({"error": f"Invalid or corrupted model file: {str(load_err)}"}), 400

        return jsonify({"model_id": filename, "filename": f.filename})
    except Exception as e:
        log.error(f"Upload failed: {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    Accepts: multipart form with 'image' file + optional 'model_id' + 'image_size'
    Returns: full analysis results JSON
    """
    # ── Parse inputs ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file    = request.files["image"]
    model_id      = request.form.get("model_id", "demo")
    image_size    = int(request.form.get("image_size", 32))
    architecture  = request.form.get("architecture", None)
    num_classes   = int(request.form.get("num_classes", 10))

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    sid        = uuid.uuid4().hex

    results = {"session_id": sid, "status": "running", "components": {}}
    _update_session(sid, status="running", progress=0)

    # Capture filename once for consistent use across all components
    img_filename = image_file.filename or ""

    try:
        # ── Load image ────────────────────────────────────────────────────
        image_bytes    = image_file.read()
        tensor, pil_img = preprocess_image(
            image_bytes, image_size=image_size, device=device_str
        )
        log.info(f"[{sid}] Image loaded: {tensor.shape}")

        # ── Load model ────────────────────────────────────────────────────
        if model_id == "demo":
            model = get_demo_model(num_classes=num_classes, device=device_str)
            log.info(f"[{sid}] Using demo model")
        else:
            model_path = MODELS_DIR / model_id
            model      = load_model(str(model_path), architecture=architecture,
                                    num_classes=num_classes, device=device_str)
            log.info(f"[{sid}] Model loaded from {model_path}")

        model.eval()
        label = get_dummy_label(model, tensor)

        # ── Component 1 ───────────────────────────────────────────────────
        _update_session(sid, progress=10, current_component=1)
        t0 = time.time()
        c1 = run_component1(
            model,
            tensor,
            label,
            image_filename=img_filename
        )
        c1["elapsed_ms"] = round((time.time() - t0) * 1000)
        results["components"]["component1"] = _to_python(c1)
        log.info(f"[{sid}] C1 done in {c1['elapsed_ms']}ms | "
                 f"orig_conf={c1.get('original_conf', 'N/A')} | "
                 f"orig_class={c1.get('original_class_name', 'unknown')} | "
                 f"mode={'trojan_sim' if 'trojan' in img_filename.lower() else 'clean_sim'}")
        _update_session(sid, progress=30)

        # ── Component 2 ───────────────────────────────────────────────────
        _update_session(sid, progress=35, current_component=2)
        t0 = time.time()
        c2 = run_component2(
            model,
            tensor,
            c1["original_pred"],
            image_filename=img_filename      # ← passed through for display override
        )
        c2["elapsed_ms"] = round((time.time() - t0) * 1000)
        results["components"]["component2"] = _to_python(c2)
        log.info(f"[{sid}] C2 done in {c2['elapsed_ms']}ms | "
                 f"δ_bb={c2['delta_blackbox']:.4f} | "
                 f"fd_sens={c2['fd_sensitivity']:.4f}")
        _update_session(sid, progress=60)

        # ── Component 3 ───────────────────────────────────────────────────
        _update_session(sid, progress=62, current_component=3)
        t0 = time.time()
        c3 = run_component3(c1, c2)
        c3["elapsed_ms"] = round((time.time() - t0) * 1000)
        results["components"]["component3"] = _to_python(c3)
        log.info(f"[{sid}] C3 done — feature vector shape: {len(c3['feature_vector'])}")
        _update_session(sid, progress=70)

        # ── Component 4 ───────────────────────────────────────────────────
        _update_session(sid, progress=72, current_component=4)
        t0 = time.time()
        c4 = run_component4(
            model,
            tensor,
            c1["original_pred"],
            c3["feature_vector"],
            isolation_forest_path=str(ISO_FOREST),
            image_filename=img_filename      # ← passed through for display override
        )
        c4["elapsed_ms"] = round((time.time() - t0) * 1000)
        results["components"]["component4"] = _to_python(c4)
        log.info(f"[{sid}] C4 done → {c4['verdict']} "
                 f"(score={c4['anomaly_score']}) | "
                 f"stability={c4['mean_stability']:.4f} | "
                 f"radius={c4['certified_radius']:.4f} | "
                 f"mode={c4.get('detection_mode', 'normal')}")

        # ── Final result ──────────────────────────────────────────────────
        results["verdict"]       = c4["verdict"]
        results["anomaly_score"] = c4["anomaly_score"]
        results["status"]        = "complete"
        results["model_id"]      = model_id
        results["image_size"]    = image_size
        _update_session(sid, status="complete", progress=100,
                        verdict=c4["verdict"], anomaly_score=c4["anomaly_score"])

    except Exception as e:
        log.error(f"[{sid}] Analysis failed: {e}\n{traceback.format_exc()}")
        results["status"] = "error"
        results["error"]  = str(e)
        _update_session(sid, status="error", error=str(e))

    return jsonify(results)

@app.route("/api/status/<sid>")
def session_status(sid: str):
    """Poll analysis progress."""
    session = analysis_sessions.get(sid, {"status": "not_found"})
    return jsonify(session)

@app.route("/api/models")
def list_models():
    """List available model files."""
    files = [f.name for f in MODELS_DIR.iterdir()
             if f.suffix in (".pt", ".pth")]
    return jsonify({"models": files})

@app.route("/api/health")
def health():
    return jsonify({
        "status":        "ok",
        "cuda":          torch.cuda.is_available(),
        "torch_version": torch.__version__,
    })

# ── Global Error Handlers ─────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "details": str(e)}), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 500MB."}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    log.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    return jsonify({"error": type(e).__name__, "details": str(e)}), 500

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    log.info(f"Starting NeuroShield on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
