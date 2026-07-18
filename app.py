import os, sys, json, io, base64
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b5
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# ── paths ──
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR   = os.path.dirname(BASE_DIR)  # D:/Codex/test

_DEFAULT_MODEL  = os.path.join(PARENT_DIR, "checkpoints", "best_model.pt")
_DEFAULT_CLASS  = "E:/pythonproject/pest/cat_to_name.json"
_FALLBACK_MODEL = "D:/Codex/test/checkpoints/best_model.pt"
_FALLBACK_CLASS = "E:/pythonproject/pest/cat_to_name.json"

MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL)
CLASS_MAP  = os.environ.get("CLASS_MAP",  _DEFAULT_CLASS)
if not os.path.exists(MODEL_PATH):
    print(f"[WARN] MODEL_PATH not found: {MODEL_PATH}")
    if os.path.exists(_FALLBACK_MODEL):
        MODEL_PATH = _FALLBACK_MODEL
if not os.path.exists(CLASS_MAP):
    print(f"[WARN] CLASS_MAP not found: {CLASS_MAP}")
    if os.path.exists(_FALLBACK_CLASS):
        CLASS_MAP = _FALLBACK_CLASS

UPLOAD_DIR    = os.path.join(BASE_DIR, "uploaded")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ── training normalisation ──
MEAN = [0.5050, 0.5589, 0.3754]
STD  = [0.1842, 0.1776, 0.1822]
SZ   = 240

val_tf = transforms.Compose([
    transforms.Resize((SZ, SZ)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── load class names ──
def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        mx = max(int(k) for k in raw)
        names = [""] * (mx + 1)
        for k, v in raw.items():
            names[int(k)] = v
        return names
    except Exception as e:
        print(f"[WARN] Could not load class map ({e}), using indices")
        return None

class_names = load_class_names(CLASS_MAP)
NUM_CLASSES = len(class_names) if class_names else 102

# ── model ──
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = efficientnet_b5(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
)

if os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading model from {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    best_acc = ckpt.get("best_acc", "N/A")
    print(f"[INFO] Model loaded. best_acc={best_acc}")
else:
    print(f"[WARN] Model not found at {MODEL_PATH}, using random weights")

model = model.to(device)
model.eval()

# ── helpers ──
def predict(image: Image.Image, top_k: int = 5):
    img = val_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, min(top_k, NUM_CLASSES))
    results = []
    for i in range(len(top_indices)):
        idx = int(top_indices[i])
        name = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
        results.append({
            "index": idx,
            "name": name,
            "probability": round(float(top_probs[i]), 4),
        })
    return results

# ── routes ──
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    img = None
    if "image" in request.files:
        f = request.files["image"]
        if f.filename:
            img = Image.open(f.stream).convert("RGB")
    elif request.is_json:
        data = request.get_json()
        b64 = data.get("image", "")
        if b64:
            raw = base64.b64decode(b64.split(",")[-1])
            img = Image.open(io.BytesIO(raw)).convert("RGB")

    if img is None:
        return jsonify({"error": "No image provided"}), 400

    top_k = request.form.get("top_k", 5, type=int)
    results = predict(img, top_k)
    return jsonify({"results": results})

@app.route("/api/info")
def api_info():
    return jsonify({
        "num_classes": NUM_CLASSES,
        "model": "EfficientNet-B5",
        "input_size": SZ,
    })

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

if __name__ == "__main__":
    print(f"[INFO] Model path: {MODEL_PATH}")
    print(f"[INFO] Class map: {CLASS_MAP}")
    app.run(host="0.0.0.0", port=5000, debug=False)
