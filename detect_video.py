# import cv2
# from ultralytics import YOLO

# # ==========================
# # LOAD MODELS
# # ==========================
# # Person detector + tracker
# person_model = YOLO("yolo11s.pt")   # use yolo11n.pt for more FPS

# # Helmet detector (trained on ppe_safe / ppe_unsafe)
# helmet_model = YOLO("/home/priyanshu/Desktop/Internship/helmet_detection/helmet-detection-yolov8/models/hemletYoloV8_100epochs.pt")
# OUTPUT_PATH = "helmet_demo_output_1.mp4"


# print("Helmet model path:", helmet_model.ckpt_path)
# print("Helmet classes:", helmet_model.names)

# # ==========================
# # VIDEO SOURCE
# # ==========================
# VIDEO_PATH = "/home/priyanshu/Desktop/Internship/helmet_detection/kursi_test.mp4"
# cap = cv2.VideoCapture(VIDEO_PATH)
# fps = cap.get(cv2.CAP_PROP_FPS)
# if fps == 0 or fps is None:
#     fps = 25
# width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# cv2.namedWindow("Person + Helmet Tracking", cv2.WINDOW_NORMAL)

# # ==========================
# # MAIN LOOP
# # ==========================
# while cap.isOpened():
#     ret, frame = cap.read()
    
#     if not ret:
#         break

#     # --------------------------------
#     # 1. PERSON DETECTION + TRACKING
#     # --------------------------------
#     person_results = person_model.track(
#         frame,
#         persist=True,
#         classes=[0],     # person only
#         conf=0.4,
#         iou=0.5
#     )[0]

#     persons = []

#     if person_results.boxes.id is not None:
#         for box, track_id in zip(
#             person_results.boxes.xyxy,
#             person_results.boxes.id
#         ):
#             x1, y1, x2, y2 = map(int, box)
#             persons.append({
#                 "id": int(track_id),
#                 "box": [x1, y1, x2, y2],
#                 "helmet": "ppe_unsafe"   # default
#             })

#     # --------------------------------
#     # 2. HELMET DETECTION (FULL FRAME)
#     # --------------------------------
#     helmet_results = helmet_model(
#         frame,
#         conf=0.1,      # IMPORTANT: helmets are small
#         imgsz=640
#     )[0]

#     helmets = []

#     if helmet_results.boxes is not None:
#         for box, cls in zip(
#             helmet_results.boxes.xyxy,
#             helmet_results.boxes.cls
#         ):
#             x1, y1, x2, y2 = map(int, box)
#             helmets.append({
#                 "box": [x1, y1, x2, y2],
#                 "label": helmet_model.names[int(cls)]
#             })

#     # --------------------------------
#     # 3. ASSOCIATE HELMET → PERSON
#     #    (CORRECT LOGIC – NO IoU)
#     # --------------------------------
#     for person in persons:
#         px1, py1, px2, py2 = person["box"]

#         # Define HEAD REGION (top 30% of person box)
#         head_y2 = py1 + int(0.30 * (py2 - py1))

#         helmet_found = False

#         for helmet in helmets:
#             hx1, hy1, hx2, hy2 = helmet["box"]

#             # Helmet center
#             cx = (hx1 + hx2) // 2
#             cy = (hy1 + hy2) // 2

#             # Check if helmet center lies inside head region
#             if px1 < cx < px2 and py1 < cy < head_y2:
#                 helmet_found = True
#                 person["helmet"] = helmet["label"]
#                 break

#         if not helmet_found:
#             person["helmet"] = "ppe_unsafe"

#     # --------------------------------
#     # 4. VISUALIZATION
#     # --------------------------------
#     for p in persons:
#         x1, y1, x2, y2 = p["box"]
#         pid = p["id"]
#         label = p["helmet"]

#         if label == "ppe_safe":
#             color = (0, 255, 0)
#         else:
#             color = (0, 0, 255)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(
#             frame,
#             f"ID {pid} | {label}",
#             (x1, max(25, y1 - 8)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             color,
#             2
#         )

#         # OPTIONAL: visualize head region (comment for demo)
#         # cv2.rectangle(frame, (x1, y1), (x2, head_y2), (255, 255, 0), 1)
#     out.write(frame)
#     cv2.imshow("Person + Helmet Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from ultralytics import YOLO
from collections import defaultdict

# ==========================
# CONFIG
# ==========================
VIDEO_PATH = 0 # "ch75m_20251118075737t_n04.mp4"
MODEL_PATH = "helmet-detection-yolov8/models/hemletYoloV8_100epochs.pt"
OUTPUT_PATH = "helmet_head_smooth_final.mp4"

CONF_THRESHOLD = 0.10      # small objects
IOU_THRESHOLD = 0.5
SMOOTH_ALPHA = 0.85        # HIGH = very stable boxes

HEAD_ID = 0
HELMET_ID = 1

# ==========================
# LOAD MODEL
# ==========================
model = YOLO(MODEL_PATH)

print("Model loaded from:", MODEL_PATH)
print("Model classes:", model.names)

# ==========================
# VIDEO SETUP
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps == 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

cv2.namedWindow("Helmet / No Helmet (Smoothed)", cv2.WINDOW_NORMAL)

# ==========================
# SMOOTHING STORAGE (PER TRACK ID)
# ==========================
smooth_boxes = defaultdict(lambda: None)

def smooth_bbox(track_id, new_box):
    if smooth_boxes[track_id] is None:
        smooth_boxes[track_id] = new_box
        return new_box

    old = smooth_boxes[track_id]
    smoothed = [
        int(SMOOTH_ALPHA * old[i] + (1 - SMOOTH_ALPHA) * new_box[i])
        for i in range(4)
    ]
    smooth_boxes[track_id] = smoothed
    return smoothed

# ==========================
# MAIN LOOP
# ==========================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # --------------------------------
    # TRACK HEAD & HELMET ONLY
    # --------------------------------
    results = model.track(
        frame,
        persist=True,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=[HEAD_ID, HELMET_ID]
    )[0]

    if results.boxes.id is not None:
        for box, cls, track_id in zip(
            results.boxes.xyxy,
            results.boxes.cls,
            results.boxes.id
        ):
            x1, y1, x2, y2 = map(int, box)
            tid = int(track_id)
            cls_id = int(cls)

            # Smooth bounding box
            x1, y1, x2, y2 = smooth_bbox(tid, [x1, y1, x2, y2])

            if cls_id == HEAD_ID:
                label = "NO HELMET"
                color = (0, 0, 255)   # 🔴 RED
            else:
                label = "HELMET"
                color = (0, 255, 0)   # 🟢 GREEN

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    out.write(frame)
    cv2.imshow("Helmet / No Helmet (Smoothed)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Periodically clear stale IDs
    if frame_count % 400 == 0:
        smooth_boxes.clear()

# ==========================
# CLEANUP
# ==========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("Output saved to:", OUTPUT_PATH)

