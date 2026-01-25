import cv2
from ultralytics import YOLO
import os
import time
import math
from collections import deque
import torch

# ==========================
# DOCKER / HEADLESS CONFIG
# ==========================
HEADLESS = True   # MUST be True for Docker / server
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# ==========================
# CONFIG
# ==========================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;0|buffer_size;102400"
)

RTSP_URL = "rtsp://admin:Visomni%402026@59.145.221.92:554/Streaming/Channels/102"

HEAD_MODEL_PATH = "hemletYoloV8_100epochs.pt"
HELMET_MODEL_PATH = "/home/priyanshu/Desktop/Internship/helmet_detection/training_best_model/runs/detect/helmet_detector/weights/best.pt"

OUTPUT_PATH = "helmet_two_stage_output_roi.mp4"

# ==========================
# ROI (FIXED FOR DOCKER)
# ==========================
# (x, y, width, height) — set once from testing
ROI = (200, 150, 800, 450)

rx1, ry1, rw, rh = ROI
rx2, ry2 = rx1 + rw, ry1 + rh

# ==========================
# PARAMETERS (UNCHANGED)
# ==========================
HEAD_CONF = 0.30
HELMET_CONF = 0.35

MATCH_DIST = 60
MAX_MISSING_TIME = 0.4

SMOOTH_ALPHA = 0.6
SNAP_MOTION = 35

MIN_VOTES = 6

# ==========================
# OUTPUT FOLDERS
# ==========================
SAVE_ROOT = "captures"
HELMET_DIR = os.path.join(SAVE_ROOT, "helmet")
NO_HELMET_DIR = os.path.join(SAVE_ROOT, "no_helmet")

os.makedirs(HELMET_DIR, exist_ok=True)
os.makedirs(NO_HELMET_DIR, exist_ok=True)

# ==========================
# LOAD MODELS
# ==========================
head_model = YOLO(HEAD_MODEL_PATH)
helmet_model = YOLO(HELMET_MODEL_PATH)

print("Head model:", head_model.names)
print("Helmet model:", helmet_model.names)

# ==========================
# VIDEO
# ==========================
def open_rtsp():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

cap = open_rtsp()

# ==========================
# VIDEO OUTPUT
# ==========================
fps = 10
W, H = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

# ==========================
# TRACK STATE
# ==========================
tracks = {}
next_id = 1

# ==========================
# HELPERS
# ==========================
def center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def inside_roi(box):
    cx, cy = center(box)
    return rx1 < cx < rx2 and ry1 < cy < ry2

def save_crop(frame, box, label, tid):
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    folder = HELMET_DIR if label == "HELMET" else NO_HELMET_DIR
    cv2.imwrite(os.path.join(folder, f"id_{tid}.jpg"), crop)

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ RTSP reconnecting...")
        cap.release()
        time.sleep(1)
        cap = open_rtsp()
        continue

    frame = cv2.resize(frame, (W, H))
    now = time.time()

    # ==========================
    # STAGE 1 — HEAD DETECTION
    # ==========================
    head_results = head_model(
        frame,
        conf=HEAD_CONF,
        iou=0.5,
        verbose=False
    )[0]

    detections = []

    for box, cls in zip(
        head_results.boxes.xyxy,
        head_results.boxes.cls
    ):
        label = head_model.names[int(cls)]
        if label not in ("head", "helmet"):
            continue

        box = list(map(int, box))
        if not inside_roi(box):
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
                "raw_center": center(box),
                "prev_box": box[:],
                "helmet_votes": deque(maxlen=20),
                "decision_ready": False,
                "final_label": None,
                "saved": False,
                "last_seen": now
            }

        t = tracks[matched]
        t["last_seen"] = now

        # ==========================
        # SNAP vs SMOOTH
        # ==========================
        prev = t["prev_box"]
        pcx, pcy = center(prev)
        ncx, ncy = center(box)
        motion = abs(pcx - ncx) + abs(pcy - ncy)

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
        x1, y1, x2, y2 = t["box"]
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            helmet_res = helmet_model(
                crop,
                conf=HELMET_CONF,
                verbose=False
            )[0]
            if helmet_res.boxes:
                cls2 = int(helmet_res.boxes.cls[0])
                helmet_label = helmet_model.names[cls2]
                t["helmet_votes"].append(helmet_label)

        # ==========================
        # FINAL DECISION (UNCHANGED)
        # ==========================
        if not t["decision_ready"] and len(t["helmet_votes"]) >= MIN_VOTES:
            no_helmet_votes = t["helmet_votes"].count("no_helmet")
            helmet_votes = t["helmet_votes"].count("helmet")

            if no_helmet_votes > 0:
                t["final_label"] = "NO_HELMET"
            elif helmet_votes == len(t["helmet_votes"]) and head_label == "helmet":
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

        if not t["saved"]:
            save_crop(frame, t["box"], label, tid)
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

    # ROI outline
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    out.write(frame)

    if not HEADLESS:
        cv2.imshow("Helmet Two-Stage Ensemble (ROI Locked)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ==========================
# CLEANUP
# ==========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Saved:", OUTPUT_PATH)
