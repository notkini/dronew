"""
plate_receiver.py — Laptop-side plate image receiver
Receives cropped plate images from the RPi via HTTP POST
and saves them into organised folders.

Folder structure created automatically:
    received_plates/
    ├── violation/      ← no-helmet rider plates
    └── general/        ← all other vehicle plates

Run on your laptop BEFORE starting detect_stream.py on the RPi:
    pip install flask
    python plate_receiver.py

The server listens on all interfaces at port 8765.
The RPi must be able to reach your laptop on this port
(check Windows Firewall / macOS firewall if needed).
"""

import os
import time
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────
# CONFIG — change port here if 8765 is blocked
# ─────────────────────────────────────────────
PORT       = 8765
SAVE_ROOT  = "received_plates"   # relative to where you run this script

VIOLATION_DIR = os.path.join(SAVE_ROOT, "violation")
GENERAL_DIR   = os.path.join(SAVE_ROOT, "general")

os.makedirs(VIOLATION_DIR, exist_ok=True)
os.makedirs(GENERAL_DIR,   exist_ok=True)

app = Flask(__name__)

# Simple counters for the session
counters = {"violation": 0, "general": 0}


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    img_file   = request.files['image']
    plate_type = request.form.get('plate_type', 'general')
    score      = request.form.get('score', '0.00')
    frame_id   = request.form.get('frame_id', '0')

    # Sanitise plate_type to avoid path traversal
    if plate_type not in ("violation", "general"):
        plate_type = "general"

    save_dir  = VIOLATION_DIR if plate_type == "violation" else GENERAL_DIR

    # Build a unique filename (timestamp + score)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename  = f"{plate_type}_{timestamp}_f{frame_id}_s{score}.jpg"
    save_path = os.path.join(save_dir, filename)

    img_file.save(save_path)
    counters[plate_type] += 1

    tag = "🚨 VIOLATION" if plate_type == "violation" else "🔵 General"
    print(f"{tag} plate saved → {save_path}  "
          f"(score={score}, frame={frame_id})  "
          f"[session total: V={counters['violation']} G={counters['general']}]")

    return jsonify({"status": "ok", "saved": save_path}), 200


@app.route('/status', methods=['GET'])
def status():
    """Quick health-check endpoint — open in browser to verify receiver is up."""
    return jsonify({
        "status":    "running",
        "violation": counters["violation"],
        "general":   counters["general"],
        "save_root": os.path.abspath(SAVE_ROOT),
    })


if __name__ == "__main__":
    print("=" * 55)
    print(" Plate Receiver — listening for RPi uploads")
    print(f" Port      : {PORT}")
    print(f" Saving to : {os.path.abspath(SAVE_ROOT)}/")
    print(f"   violation/  ← no-helmet rider plates")
    print(f"   general/    ← all other plates")
    print(f" Health check: http://localhost:{PORT}/status")
    print("=" * 55)
    print()
    app.run(host='0.0.0.0', port=PORT, threaded=True)
