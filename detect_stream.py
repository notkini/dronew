"""
detect_stream.py — YOLOv8n Coral TPU + Flask MJPEG Stream
Features:
  - Live webcam inference on Coral TPU
  - Spatial association of plates to riders
  - Only violation plates get flagged
  - Stream viewable on laptop browser

Run:
    source ~/coral-env/bin/activate
    cd ~/traffic_drone
    python detect_stream.py

View on laptop:
    http://192.168.0.10:5000
"""

import time
import threading
import numpy as np
import cv2
from flask import Flask, Response, render_template_string
from pycoral.utils.edgetpu import make_interpreter

MODEL      = "models/best_full_int8_edgetpu.tflite"
CLASSES    = ["helmet", "no_helmet", "rider", "plate"]
THRESHOLD  = 0.20
INPUT_SIZE = 320
CAMERA_ID  = 0

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
            "label": CLASSES[class_ids[i]],
            "score": round(float(confidences[i]), 3),
            "bbox":  (int(x1*orig_w), int(y1*orig_h),
                      int(x2*orig_w), int(y2*orig_h)),
            "violation_plate": False
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
        results = [r for r in results if iou(best["bbox"], r["bbox"]) < iou_threshold]
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
    Returns list of violation plate detections.
    """
    riders     = [d for d in detections if d["label"] in ("rider", "no_helmet", "helmet")]
    plates     = [d for d in detections if d["label"] == "plate"]
    no_helmets = [d for d in detections if d["label"] == "no_helmet"]

    violation_plates = []

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

        if best_rider is None:
            continue

        # Direct no_helmet detection matched to this plate
        if best_rider["label"] == "no_helmet":
            plate["violation_plate"] = True
            violation_plates.append(plate)

        elif best_rider["label"] == "rider":
            # Check if any no_helmet overlaps with this rider
            for nh in no_helmets:
                if horizontal_overlap(nh["bbox"], best_rider["bbox"]):
                    plate["violation_plate"] = True
                    violation_plates.append(plate)
                    break

    return violation_plates


# ─────────────────────────────────────────────
# ANNOTATION
# ─────────────────────────────────────────────

def annotate_frame(frame, detections, violations, violation_plates, fps, ms):
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]

        # Use red for violation plates, normal color otherwise
        if d["label"] == "plate" and d.get("violation_plate"):
            color = COLORS["plate_vio"]
        else:
            color = COLORS.get(d["label"], (180, 180, 180))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{d['label']} {d['score']:.2f}"
        if d["label"] == "plate" and d.get("violation_plate"):
            label_text = f"VIOLATION PLATE {d['score']:.2f}"
        cv2.rectangle(frame, (x1, y1-22), (x1+len(label_text)*9, y1), color, -1)
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

    # Violation plate count
    if violation_plates:
        cv2.putText(frame, f"Violation plates: {len(violation_plates)}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

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
        h1 { color: #00ff88; margin: 20px; }
        img { border: 2px solid #00ff88; border-radius: 8px; max-width: 100%; }
        p { color: #aaa; }
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
    fps = 0
    fps_timer = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        orig_h, orig_w = frame.shape[:2]

        # Preprocess
        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        inp     = np.expand_dims(rgb.astype(np.uint8), axis=0)

        # Inference
        interpreter.set_tensor(input_details[0]['index'], inp)
        t0 = time.perf_counter()
        interpreter.invoke()
        ms = (time.perf_counter() - t0) * 1000

        # Output
        raw = interpreter.get_tensor(output_details[0]['index'])
        scale, zero_point = output_details[0]['quantization']
        if scale != 0:
            raw = (raw.astype(np.float32) - zero_point) * scale
        else:
            raw = raw.astype(np.float32)

        detections = parse_output(raw, orig_w, orig_h, THRESHOLD)

        # Spatial association
        violation_plates = associate_plates_to_riders(detections)

        # Violation logic
        labels_found = {d["label"] for d in detections}
        violations = []
        if "no_helmet" in labels_found:
            violations.append("HELMET_VIOLATION")

        # FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_timer)
            fps_timer = time.time()

        # Annotate
        annotated = annotate_frame(frame, detections, violations,
                                   violation_plates, fps, ms)

        # Terminal log every 30 frames
        if frame_count % 30 == 0:
            print(f"Frame {frame_count} | {ms:.0f}ms | FPS:{fps:.1f} | "
                  f"Objects:{len(detections)} | ViolationPlates:{len(violation_plates)}")

        # Update shared frame
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
    print("Open on your laptop: http://192.168.0.10:5000")
    print("Press Ctrl+C to stop\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)
