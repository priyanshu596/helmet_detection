import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from collections import deque

# ==========================
# CONFIG
# ==========================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;102400|max_delay;0"
)

RTSP_URL = "rtsp://admin:Visomni%402026@59.145.221.92:554/Streaming/Channels/102"

HEAD_MODEL_PATH = "/home/priyanshu/Desktop/Internship/helmet_detection_final_high_acuuracy/hemletYoloV8_100epochs.pt"
HELMET_MODEL_PATH = "/home/priyanshu/Desktop/Internship/helmet_detection_final_high_acuuracy/best.pt"

OUTPUT_PATH = "helmet_bytetrack_roi_output.mp4"

# ==========================
# PARAMETERS
# ==========================
HEAD_CONF = 0.35
HELMET_CONF = 0.30

MIN_VOTES = 6
NO_HELMET_MIN_VOTES = 2   # correction to avoid single-frame false negative

MAX_MISSING_FRAMES = 15

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
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps = 10
W, H = 1280, 720
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

# ==========================
# ROI STATE
# ==========================
roi_points = []
roi_ready = False

counted_helmet_ids = set()
counted_no_helmet_ids = set()

# ==========================
# TRACK STATE (ByteTrack ID)
# ==========================
tracks = {}
frame_idx = 0

# Each track:
# {
#   box,
#   helmet_votes [(label, conf)],
#   decision_ready,
#   final_label,
#   last_seen_frame
# }

# ==========================
# ROI HELPERS
# ==========================
def draw_roi(event, x, y, flags, param):
    global roi_points, roi_ready
    if roi_ready:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points.append((x, y))

def inside_roi(point, roi):
    if not roi_ready:
        return False
    return cv2.pointPolygonTest(
        np.array(roi, np.int32),
        point,
        False
    ) >= 0

# ==========================
# WINDOW
# ==========================
cv2.namedWindow("Helmet PPE (ROI + ByteTrack)")
cv2.setMouseCallback("Helmet PPE (ROI + ByteTrack)", draw_roi)

# ==========================
# MAIN LOOP
# ==========================
while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(1)
        continue

    frame = cv2.resize(frame, (W, H))
    frame_idx += 1

    # ==========================
    # STAGE 1 — HEAD DETECTION + BYTETRACK
    # ==========================
    results = head_model.track(
        frame,
        conf=HEAD_CONF,
        iou=0.5,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )[0]

    if results.boxes is not None and results.boxes.id is not None:
        for box, cls, tid, conf in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.id,
            results.boxes.conf
        ):
            track_id = int(tid)
            head_label = head_model.names[int(cls)]

            # Only head / helmet drive tracking
            if head_label not in ("head", "helmet"):
                continue

            x1, y1, x2, y2 = map(int, box)

            if track_id not in tracks:
                tracks[track_id] = {
                    "box": [x1, y1, x2, y2],
                    "helmet_votes": deque(maxlen=20),
                    "decision_ready": False,
                    "final_label": None,
                    "last_seen_frame": frame_idx
                }

            t = tracks[track_id]
            t["box"] = [x1, y1, x2, y2]
            t["last_seen_frame"] = frame_idx

            # ==========================
            # STAGE 2 — HELMET CLASSIFIER
            # ==========================
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                helmet_res = helmet_model(
                    crop,
                    conf=HELMET_CONF,
                    verbose=False
                )[0]

                if helmet_res.boxes:
                    cls2 = int(helmet_res.boxes.cls[0])
                    h_label = helmet_model.names[cls2]
                    h_conf = float(helmet_res.boxes.conf[0])

                    # Ignore very weak no_helmet
                    if h_label == "no_helmet" and h_conf < 0.45:
                        pass
                    else:
                        t["helmet_votes"].append((h_label, h_conf))

            # ==========================
            # FINAL DECISION (CORRECTED)
            # ==========================
            if not t["decision_ready"] and len(t["helmet_votes"]) >= MIN_VOTES:
                no_votes = [(c) for l, c in t["helmet_votes"] if l == "no_helmet"]
                h_votes = [(c) for l, c in t["helmet_votes"] if l == "helmet"]

                no_score = sum(no_votes)
                h_score = sum(h_votes)

                if len(no_votes) >= NO_HELMET_MIN_VOTES and no_score > h_score:
                    t["final_label"] = "NO_HELMET"
                elif h_score * 1.3 > no_score:
                    t["final_label"] = "HELMET"
                else:
                    t["final_label"] = "NO_HELMET"

                t["decision_ready"] = True

    # ==========================
    # DRAW ROI
    # ==========================
    if len(roi_points) > 1:
        cv2.polylines(
            frame,
            [np.array(roi_points, np.int32)],
            isClosed=roi_ready,
            color=(255, 255, 0),
            thickness=2
        )

    for p in roi_points:
        cv2.circle(frame, p, 4, (0, 255, 255), -1)

    # ==========================
    # DRAW + COUNT + CLEANUP
    # ==========================
    for tid in list(tracks.keys()):
        t = tracks[tid]

        if frame_idx - t["last_seen_frame"] > MAX_MISSING_FRAMES:
            del tracks[tid]
            continue

        if not t["decision_ready"]:
            continue

        x1, y1, x2, y2 = t["box"]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
