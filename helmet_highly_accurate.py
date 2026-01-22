import cv2
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

RTSP_URL = "/home/priyanshu/Desktop/Internship/recording.mp4"#"rtsp://admin:Visomni%402026@59.145.221.92:554/Streaming/Channels/102"

HEAD_MODEL_PATH = "best.pt"
HELMET_MODEL_PATH = "yolo_v8_separately.pt"

OUTPUT_PATH = "helmet_two_stage_bytetrack_balanced.mp4"

# ==========================
# PARAMETERS
# ==========================
HEAD_CONF = 0.35
HELMET_CONF = 0.30

# Decision delay
MIN_VOTES = 6

# Track cleanup
MAX_MISSING_FRAMES = 15   # ByteTrack already handles most cases

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
# TRACK STATE (BYTETRACK ID)
# ==========================
tracks = {}

# Each track_id:
# {
#   box,
#   helmet_votes (deque),
#   decision_ready,
#   final_label,
#   last_seen_frame
# }

frame_idx = 0

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

    if results.boxes is None or results.boxes.id is None:
        out.write(frame)
        cv2.imshow("Helmet Two-Stage (ByteTrack)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    for box, cls, tid in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.id
    ):
        track_id = int(tid)
        label_head = head_model.names[int(cls)]

        # Only head or helmet boxes drive tracking
        if label_head not in ("head", "helmet"):
            continue

        x1, y1, x2, y2 = map(int, box)

        # Initialize track if new
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
                helmet_label = helmet_model.names[cls2]
                t["helmet_votes"].append(helmet_label)

        # ==========================
        # FINAL DECISION (ENSEMBLE)
        # ==========================
        if not t["decision_ready"] and len(t["helmet_votes"]) >= MIN_VOTES:
            no_helmet_votes = t["helmet_votes"].count("no_helmet")
            helmet_votes = t["helmet_votes"].count("helmet")

            # 🔒 SAFETY-FIRST RULE
            if no_helmet_votes > 0:
                t["final_label"] = "NO_HELMET"
            elif helmet_votes == len(t["helmet_votes"]) and label_head == "helmet":
                t["final_label"] = "HELMET"
            else:
                t["final_label"] = "NO_HELMET"

            t["decision_ready"] = True

    # ==========================
    # DRAW + CLEANUP
    # ==========================
    for tid in list(tracks.keys()):
        t = tracks[tid]

        # Remove stale tracks
        if frame_idx - t["last_seen_frame"] > MAX_MISSING_FRAMES:
            del tracks[tid]
            continue

        if not t["decision_ready"]:
            continue  # intentional delay, no CHECKING

        x1, y1, x2, y2 = t["box"]
        label = t["final_label"]

        color = (0, 0, 255) if label == "NO_HELMET" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    out.write(frame)
    cv2.imshow("Helmet Two-Stage (ByteTrack)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==========================
# CLEANUP
# ==========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Saved:", OUTPUT_PATH)
