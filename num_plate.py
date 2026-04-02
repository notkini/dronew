"""
detect_stream.py — YOLOv8n Coral TPU + Flask MJPEG Stream
Features:
  - Live webcam inference on Coral TPU
  - Spatial association of plates to riders
  - Violation plates (no_helmet rider) flagged and cropped
  - All other plates cropped and tagged as 'general'
  - Cropped plate images sent via HTTP POST to laptop receiver
  - Cooldown deduplication so same plate isn't spammed every frame
  - Stream viewable on laptop browser

Run:
    source ~/coral-env/bin/activate
    cd ~/traffic_drone
    python detect_stream.py

View stream on laptop:
    http://192.168.0.10:5000

Plate receiver must be running on laptop:
    python plate_receiver.py
"""

import time
import threading
import numpy as np
import cv2
import requests
import io
from flask import Flask, Response, render_template_string
from pycoral.utils.edgetpu import make_interpreter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL      = "models/best_full_int8_edgetpu.tflite"
CLASSES    = ["helmet", "no_helmet", "rider", "plate"]
THRESHOLD  = 0.20
INPUT_SIZE = 320
CAMERA_ID  = 0

# ── Laptop receiver ──────────────────────────
# Set this to your laptop's IP on the same network
LAPTOP_IP            = "192.168.0.100"
LAPTOP_RECEIVER_PORT = 8765
LAPTOP_RECEIVER_URL  = f"http://{LAPTOP_IP}:{LAPTOP_RECEIVER_PORT}/upload"

# How many seconds to wait before re-sending a plate
# that was detected from the same region of the frame
PLATE_COOLDOWN_SECS = 5.0

# Minimum pixel area for a plate crop to be worth sending
MIN_PLATE_AREA = 400   # 20×20 px minimum

COLORS = {
    "helmet":    (0, 200, 0),
    "no_helmet": (0, 0, 255),
    "rider":     (0, 140, 255),
    "plate":     (0, 220, 255),
    "plate_vio": (0, 0, 255),   # violation plate — red
}

app = Flask(__name__)

# Shared frame buffer
output_frame = None
frame_lock   = threading.Lock()

# ─────────────────────────────────────────────
# PLATE DEDUPLICATION
# Tracks (cx_bucket, cy_bucket) → last_sent_time
# so we don't flood the laptop with the same plate
# ─────────────────────────────────────────────

plate_cooldown_map = {}
cooldown_lock      = threading.Lock()
BUCKET_SIZE        = 40   # pixels — plates within this radius are treated as same plate


def _bucket(bbox):
    """Convert bbox to a coarse grid cell for dedup."""
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) // 2) // BUCKET_SIZE
    cy = ((y1 + y2) // 2) // BUCKET_SIZE
    return (cx, cy)


def is_plate_on_cooldown(bbox):
    key = _bucket(bbox)
    now = time.time()
    with cooldown_lock:
        last = plate_cooldown_map.get(key, 0)
        if now - last < PLATE_COOLDOWN_SECS:
            return True
        plate_cooldown_map[key] = now
        return False


# ─────────────────────────────────────────────
# HTTP SENDER  (runs in a daemon thread pool)
# ─────────────────────────────────────────────

def send_plate_async(crop_bgr, plate_type, score, frame_id):
    """
    Encode crop as JPEG and POST to the laptop receiver.
    plate_type: "violation" | "general"
    Runs in a background thread so it never blocks inference.
    """
    def _send():
        try:
            ret, buf = cv2.imencode('.jpg', crop_bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ret:
                return
            img_bytes = buf.tobytes()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename  = f"{plate_type}_{timestamp}_f{frame_id}_s{score:.2f}.jpg"
            resp = requests.post(
                LAPTOP_RECEIVER_URL,
                files={"image": (filename, img_bytes, "image/jpeg")},
                data={"plate_type": plate_type, "score": str(score),
                      "frame_id": str(frame_id)},
                timeout=3
            )
            if resp.status_code == 200:
                print(f"  [SENT] {plate_type} plate → laptop ({filename})")
            else:
                print(f"  [WARN] Receiver returned {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print("  [WARN] Laptop receiver not reachable — is plate_receiver.py running?")
        except Exception as e:
            print(f"  [WARN] Send failed: {e}")

    t = threading.Thread(target=_send, daemon=True)
    t.start()


# ─────────────────────────────────────────────
# PLATE CROPPING
# ─────────────────────────────────────────────

def crop_plate(frame, bbox, padding=4):
    """Return a cropped plate image with a small padding border."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    area = (x2 - x1) * (y2 - y1)
    if area < MIN_PLATE_AREA:
        return None
    return frame[y1:y2, x1:x2].copy()


# ─────────────────────────────────────────────
# DETECTION HELPERS
# ─────────────────────────────────────────────

def parse_output(output_tensor, orig_w, orig_h, threshold):
    preds       = output_tensor[0].T
    boxes_xywh  = preds[:, :4]
    cls_scores  = preds[:, 4:]
    class_ids   = np.argmax(cls_scores, axis=1)
    confidences = np.max(cls_scores, axis=1)
    mask        = confidences >= threshold
    if not mask.any():
        return []
    boxes_xywh  = boxes_xywh[mask]
    class_ids   = class_ids[mask]
    confidences = confidences[mask]
    results = []
    for i in range(len(class_ids)):
        cx, cy, w, h = boxes_xywh[i]
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(1.0, cx + w / 2)
        y2 = min(1.0, cy + h / 2)
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({
            "label":           CLASSES[class_ids[i]],
            "score":           round(float(confidences[i]), 3),
            "bbox":            (int(x1*orig_w), int(y1*orig_h),
                                int(x2*orig_w), int(y2*orig_h)),
            "violation_plate": False,
            "plate_sent":      False,   # track whether we've queued a send this frame
        })
    return nms(results)


def iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union>0 else 0


def nms(results, iou_threshold=0.45):
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    kept = []
    while results:
        best = results.pop(0)
        kept.append(best)
        results = [r for r in results
                   if iou(best["bbox"], r["bbox"]) < iou_threshold]
    return kept


def horizontal_overlap(box_a, box_b):
    ax1,_,ax2,_ = box_a
    bx1,_,bx2,_ = box_b
    return ax1 < bx2 and ax2 > bx1


# ─────────────────────────────────────────────
# SPATIAL VIOLATION ASSOCIATION
# ─────────────────────────────────────────────

def associate_plates_to_riders(detections):
    """
    Match each plate to the horizontally nearest rider.
    Flag plate as violation if its matched rider has no_helmet.
    Returns (violation_plates, general_plates).

    - violation_plates : plates linked to a no_helmet rider
    - general_plates   : plates linked to a helmeted rider OR orphan plates
                         (no nearby rider found) — still worth capturing
    """
    riders     = [d for d in detections if d["label"] in ("rider", "no_helmet", "helmet")]
    plates     = [d for d in detections if d["label"] == "plate"]
    no_helmets = [d for d in detections if d["label"] == "no_helmet"]

    violation_plates = []
    general_plates   = []

    for plate in plates:
        px1, py1, px2, py2 = plate["bbox"]
        plate_cx = (px1 + px2) / 2

        # Find closest rider horizontally
        best_rider = None
        best_dist  = float('inf')
        for rider in riders:
            rx1, ry1, rx2, ry2 = rider["bbox"]
            rider_cx = (rx1 + rx2) / 2
            dist = abs(plate_cx - rider_cx)
            if dist < best_dist:
                best_dist  = dist
                best_rider = rider

        is_violation = False

        if best_rider is not None:
            if best_rider["label"] == "no_helmet":
                is_violation = True
            elif best_rider["label"] == "rider":
                for nh in no_helmets:
                    if horizontal_overlap(nh["bbox"], best_rider["bbox"]):
                        is_violation = True
                        break

        plate["violation_plate"] = is_violation

        if is_violation:
            violation_plates.append(plate)
        else:
            # Includes helmeted riders AND orphan plates (other vehicles, etc.)
            general_plates.append(plate)

    return violation_plates, general_plates


# ─────────────────────────────────────────────
# PLATE DISPATCH  (crop + send if not on cooldown)
# ─────────────────────────────────────────────

def dispatch_plates(frame, violation_plates, general_plates, frame_id):
    """Crop each plate and fire off a background send if cooldown has elapsed."""
    for plate in violation_plates:
        if is_plate_on_cooldown(plate["bbox"]):
            continue
        crop = crop_plate(frame, plate["bbox"])
        if crop is not None:
            send_plate_async(crop, "violation", plate["score"], frame_id)

    for plate in general_plates:
        if is_plate_on_cooldown(plate["bbox"]):
            continue
        crop = crop_plate(frame, plate["bbox"])
        if crop is not None:
            send_plate_async(crop, "general", plate["score"], frame_id)


# ─────────────────────────────────────────────
# ANNOTATION
# ─────────────────────────────────────────────

def annotate_frame(frame, detections, violations, violation_plates, general_plates, fps, ms):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        if d["label"] == "plate" and d.get("violation_plate"):
            color      = COLORS["plate_vio"]
            label_text = f"VIOLATION PLATE {d['score']:.2f}"
        else:
            color      = COLORS.get(d["label"], (180, 180, 180))
            label_text = f"{d['label']} {d['score']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1-22),
                      (x1 + len(label_text)*9, y1), color, -1)
        cv2.putText(frame, label_text, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Status banner
    if violations:
        cv2.rectangle(frame, (0,0), (frame.shape[1], 35), (0,0,200), -1)
        cv2.putText(frame, f"VIOLATION: {' | '.join(violations)}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    else:
        cv2.rectangle(frame, (0,0), (frame.shape[1], 35), (0,150,0), -1)
        cv2.putText(frame, "NO VIOLATION DETECTED",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    # Plate counters
    y_off = 55
    if violation_plates:
        cv2.putText(frame, f"Violation plates: {len(violation_plates)}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        y_off += 20
    if general_plates:
        cv2.putText(frame, f"General plates:   {len(general_plates)}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,220,255), 1)

    # FPS
    cv2.putText(frame, f"FPS:{fps:.1f} | {ms:.0f}ms",
                (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    return frame


# ─────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Violation Detection</title>
    <style>
        body { background: #111; color: white; font-family: Arial; text-align: center; }
        h1   { color: #00ff88; margin: 20px; }
        img  { border: 2px solid #00ff88; border-radius: 8px; max-width: 100%; }
        p    { color: #aaa; }
    </style>
</head>
<body>
    <h1>🚦 Traffic Violation Detection</h1>
    <img src="/video_feed" />
    <p>Live feed from Coral TPU — refresh if stream stops</p>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')
        time.sleep(0.05)


# ─────────────────────────────────────────────
# INFERENCE LOOP
# ─────────────────────────────────────────────

def inference_loop():
    global output_frame

    print("Loading model onto Coral TPU...")
    interpreter = make_interpreter(MODEL)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Model loaded OK\n")

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {CAMERA_ID}")
        return

    frame_count = 0
    fps         = 0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orig_h, orig_w = frame.shape[:2]

        # ── Preprocess ──────────────────────────────
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp     = np.expand_dims(rgb.astype(np.uint8), axis=0)

        # ── Inference ───────────────────────────────
        interpreter.set_tensor(input_details[0]['index'], inp)
        t0 = time.perf_counter()
        interpreter.invoke()
        ms = (time.perf_counter() - t0) * 1000

        # ── Dequantise output ────────────────────────
        raw = interpreter.get_tensor(output_details[0]['index'])
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:
            raw = (raw.astype(np.float32) - zero_point) * scale
        else:
            raw = raw.astype(np.float32)

        detections = parse_output(raw, orig_w, orig_h, THRESHOLD)

        # ── Spatial association ──────────────────────
        violation_plates, general_plates = associate_plates_to_riders(detections)

        # ── Crop & send plates to laptop ─────────────
        frame_count += 1
        dispatch_plates(frame, violation_plates, general_plates, frame_count)

        # ── Violation flags for banner ───────────────
        labels_found = {d["label"] for d in detections}
        violations   = []
        if "no_helmet" in labels_found:
            violations.append("HELMET_VIOLATION")

        # ── FPS counter ──────────────────────────────
        if frame_count % 10 == 0:
            fps       = 10 / (time.time() - fps_timer)
            fps_timer = time.time()

        # ── Annotate ─────────────────────────────────
        annotated = annotate_frame(frame, detections, violations,
                                   violation_plates, general_plates, fps, ms)

        # ── Terminal log every 30 frames ─────────────
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | {ms:.0f}ms | FPS:{fps:.1f} | "
                  f"Objects:{len(detections)} | "
                  f"VioPlates:{len(violation_plates)} | "
                  f"GenPlates:{len(general_plates)}")

        # ── Update shared frame buffer ────────────────
        with frame_lock:
            output_frame = annotated.copy()

    cap.release()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting inference thread...")
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    print("Starting Flask stream server...")
    print(f"Stream  → http://192.168.0.10:5000")
    print(f"Sending plates to → {LAPTOP_RECEIVER_URL}")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
