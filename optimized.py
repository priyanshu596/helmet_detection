import cv2
from ultralytics import YOLO
import os
import time
import math
from collections import deque
import torch
import threading
from flask import Flask, Response

# ==========================
# SYSTEM OPTIMIZATION (CPU SAFE)
# ==========================
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;discardcorrupt|flags;low_delay|buffer_size;102400|max_delay;0"
)

# ==========================
# CONFIG
# ==========================
RTSP_URL = "rtsp://admin:Visomni%402026@59.145.221.92:554/Streaming/Channels/102"

HEAD_MODEL_PATH = "models/hemletYoloV8_100epochs.pt"
HELMET_MODEL_PATH = "models/helmet_detector_best.pt"

OUTPUT_PATH = "helmet_output.mp4"

HEAD_CONF = 0.30
HELMET_CONF = 0.35

MATCH_DIST = 60
MAX_MISSING_TIME = 0.25

SMOOTH_ALPHA = 0.6
SNAP_MOTION = 35

MIN_VOTES = 6
MIN_BOX_AREA = 8000

FRAME_SKIP = 2
HELMET_INTERVAL = 3

# ==========================
# OUTPUT DIRS
# ==========================
SAVE_ROOT = "captures"
NO_HELMET_DIR = os.path.join(SAVE_ROOT, "no_helmet")
os.makedirs(NO_HELMET_DIR, exist_ok=True)

# ==========================
# LOAD MODELS (DO NOT USE .half())
# ==========================
head_model = YOLO(HEAD_MODEL_PATH)
helmet_model = YOLO(HELMET_MODEL_PATH)

print("Head model:", head_model.names)
print("Helmet model:", helmet_model.names)

# ==========================
# VIDEO INPUT
# ==========================
def open_rtsp():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap

cap = open_rtsp()

# ==========================
# VIDEO OUTPUT
# ==========================
fps = 10
W, H = 1280, 720
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

# ==========================
# TRACK STATE
# ==========================
tracks = {}
next_id = 1

# ==========================
# FLASK STREAM
# ==========================
app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()

def generate():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode(
                ".jpg",
                output_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 65]
            )
        if not ret:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def start_server():
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=True
    )

threading.Thread(target=start_server, daemon=True).start()

# ==========================
# HELPERS
# ==========================
def center(b):
    return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def save_crop_async(frame, box, tid):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size:
        cv2.imwrite(
            os.path.join(NO_HELMET_DIR, f"{int(time.time())}_{tid}.jpg"),
            crop
        )

# ==========================
# MAIN LOOP
# ==========================
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        cap.release()
        time.sleep(0.5)
        cap = open_rtsp()
        continue

    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (W, H))
    now = time.time()

    # ==========================
    # STAGE 1 — HEAD DETECTION
    # ==========================
    results = head_model(frame, conf=HEAD_CONF, iou=0.5, verbose=False)[0]
    detections = []

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        label = head_model.names[int(cls)]
        if label not in ("head", "helmet"):
            continue

        box = list(map(int, box))
        if (box[2] - box[0]) * (box[3] - box[1]) < MIN_BOX_AREA:
            continue

        detections.append((label, box))

    # ==========================
    # ASSOCIATE + TRACK
    # ==========================
    for head_label, box in detections:
        cx, cy = center(box)
        matched = None

        for tid, t in tracks.items():
            if dist(t["raw_center"], (cx, cy)) < MATCH_DIST:
                matched = tid
                break

        if matched is None:
            matched = next_id
            next_id += 1
            tracks[matched] = {
                "box": box[:],
                "prev_box": box[:],
                "raw_center": center(box),
                "helmet_votes": deque(maxlen=20),
                "decision_ready": False,
                "final_label": None,
                "saved": False,
                "last_seen": now,
                "frame_count": 0
            }

        t = tracks[matched]
        t["last_seen"] = now
        t["frame_count"] += 1

        prev = t["prev_box"]
        motion = abs(prev[0] - box[0]) + abs(prev[1] - box[1])

        if motion > SNAP_MOTION:
            t["box"] = box[:]
        else:
            t["box"] = [
                int(SMOOTH_ALPHA * t["box"][i] + (1 - SMOOTH_ALPHA) * box[i])
                for i in range(4)
            ]

        t["prev_box"] = box[:]
        t["raw_center"] = center(box)

        # ==========================
        # STAGE 2 — HELMET CLASSIFIER
        # ==========================
        if t["frame_count"] % HELMET_INTERVAL == 0:
            x1, y1, x2, y2 = t["box"]
            crop = frame[y1:y2, x1:x2]
            if crop.size:
                res = helmet_model(crop, conf=HELMET_CONF, verbose=False)[0]
                if res.boxes:
                    cls2 = int(res.boxes.cls[0])
                    t["helmet_votes"].append(helmet_model.names[cls2])

        # ==========================
        # FINAL DECISION
        # ==========================
        if not t["decision_ready"] and len(t["helmet_votes"]) >= MIN_VOTES:
            no_v = t["helmet_votes"].count("no_helmet")
            h_v = t["helmet_votes"].count("helmet")

            if no_v > 0:
                t["final_label"] = "NO_HELMET"
            elif h_v == len(t["helmet_votes"]) and head_label == "helmet":
                t["final_label"] = "HELMET"
            else:
                t["final_label"] = "NO_HELMET"

            t["decision_ready"] = True

    # ==========================
    # DRAW + SAVE
    # ==========================
    for tid in list(tracks.keys()):
        t = tracks[tid]
        if now - t["last_seen"] > MAX_MISSING_TIME:
            del tracks[tid]
            continue

        if not t["decision_ready"]:
            continue

        x1, y1, x2, y2 = t["box"]
        label = t["final_label"]

        if label == "NO_HELMET" and not t["saved"]:
            threading.Thread(
                target=save_crop_async,
                args=(frame, t["box"], tid),
                daemon=True
            ).start()
            t["saved"] = True

        color = (0, 0, 255) if label == "NO_HELMET" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out.write(frame)

    if frame_idx % 3 == 0:
        with frame_lock:
            output_frame = frame.copy()
