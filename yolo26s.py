import cv2
import time
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict, Counter

vehicle_model = YOLO("/home/merve/vehicle_plate_project/models/yolo26s.pt")
plate_model   = YOLO("/home/merve/vehicle_plate_project/models/license_plate_detector.pt")
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

names = vehicle_model.names
if isinstance(names, dict):
    name_list = [names[i] for i in sorted(names.keys())]
else:
    name_list = list(names)

wanted = {"car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "van"}
TARGET_CLASSES = [i for i, n in enumerate(name_list) if n.lower() in wanted]
if len(TARGET_CLASSES) == 0:
    TARGET_CLASSES = [3,4,5,8]
print("✅ TARGET_CLASSES:", TARGET_CLASSES)

plate_pattern = re.compile(r'^[A-Z0-9]{5,8}$')

VIDEO_PATH  = "/home/merve/vehicle_plate_project/License Plate Detection Test.mp4"
OUTPUT_PATH = "stable_anpr_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

ROI_SIZE = 640
roi_w = min(ROI_SIZE, width)
roi_h = min(ROI_SIZE, height)
x_min = max(0, (width  - roi_w) // 2)
y_min = max(0, (height - roi_h) // 2)
x_max = x_min + roi_w
y_max = y_min + roi_h

def draw_corner_box(img, x1, y1, x2, y2, color=(0,255,0), thickness=3, length=25):
    cv2.line(img,(x1,y1),(x1+length,y1),color,thickness)
    cv2.line(img,(x1,y1),(x1,y1+length),color,thickness)
    cv2.line(img,(x2,y1),(x2-length,y1),color,thickness)
    cv2.line(img,(x2,y1),(x2,y1+length),color,thickness)
    cv2.line(img,(x1,y2),(x1+length,y2),color,thickness)
    cv2.line(img,(x1,y2),(x1,y2-length),color,thickness)
    cv2.line(img,(x2,y2),(x2-length,y2),color,thickness)
    cv2.line(img,(x2,y2),(x2,y2-length),color,thickness)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    return inter / (area_a + area_b - inter + 1e-6)

# cache (crop sabit kalacak!)
plate_cache = {}                   # tid -> {"text","crop","last_frame"}
plate_history = defaultdict(list)
CONSENSUS_FRAMES = 3

frame_idx = 0
last_seen = {}
MAX_MISSED = 25

PLATE_EVERY_N = 1
PLATE_CONF_MIN = 0.60
IOU_MIN_PLATE_TO_REGION = 0.10

start_video_time = time.time()
MAX_SECONDS = 130

last_plate_boxes_global = []  # [(x1,y1,x2,y2,conf), ...]

print("🚀 STABLE ANPR STARTED")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if time.time() - start_video_time >= MAX_SECONDS:
        break

    frame_idx += 1
    start_time = time.time()

    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    draw_corner_box(frame, x_min, y_min, x_max, y_max, color=(0,0,255), thickness=3, length=35)

    roi_crop = frame[y_min:y_max, x_min:x_max]

    results = vehicle_model.track(
        roi_crop,
        persist=True,
        verbose=False,
        imgsz=640,
        classes=TARGET_CLASSES,
        conf=0.35,
        iou=0.50,
        tracker="bytetrack.yaml"
    )[0]

    dead_ids = [tid for tid, f in last_seen.items() if frame_idx - f > MAX_MISSED]
    for tid in dead_ids:
        last_seen.pop(tid, None)
        plate_cache.pop(tid, None)
        plate_history.pop(tid, None)

    # Plakaları ROI içinde bul (her N frame)
    if frame_idx % PLATE_EVERY_N == 0:
        last_plate_boxes_global = []
        pr = plate_model(roi_crop, verbose=False)[0]
        if pr.boxes is not None and len(pr.boxes) > 0:
            for pbox in pr.boxes:
                confp = float(pbox.conf[0])
                if confp < PLATE_CONF_MIN:
                    continue
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                last_plate_boxes_global.append((px1 + x_min, py1 + y_min, px2 + x_min, py2 + y_min, confp))

    if results.boxes.id is None:
        fps = 1 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        out.write(frame)
        cv2.imshow("STABLE ANPR", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    ids   = results.boxes.id.cpu().numpy()

    for box, track_id in zip(boxes, ids):
        tid = int(track_id)

        x1, y1, x2, y2 = map(int, box)
        gx1, gy1, gx2, gy2 = x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min

        vehicle_w = gx2 - gx1
        vehicle_h = gy2 - gy1
        if vehicle_w * vehicle_h < 25000:
            continue

        last_seen[tid] = frame_idx
        draw_corner_box(frame, gx1, gy1, gx2, gy2)

        # Panel çizimi (cache varsa direkt çiz — crop sabit)
        if tid in plate_cache:
            data = plate_cache[tid]
            text = data["text"]
            plate_resized = cv2.resize(data["crop"], (220, 70))

            panel_x = gx1
            panel_y = gy1 - 70 - 60
            if panel_y < 0:
                panel_y = gy2 + 10

            panel_x = max(0, min(panel_x, width - 220))
            panel_y = max(0, min(panel_y, height - 70))

            frame[panel_y:panel_y+70, panel_x:panel_x+220] = plate_resized
            cv2.rectangle(frame, (panel_x, panel_y-40), (panel_x+220, panel_y), (255,255,255), -1)
            cv2.putText(frame, text, (panel_x+20, panel_y-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            continue  # ✅ çok önemli: cache varsa OCR'a hiç girme

        # ---- cache yoksa: plaka eşle + OCR ----
        region_y1 = gy1 + int(vehicle_h * 0.65)
        region_y2 = gy1 + int(vehicle_h * 0.98)
        region = (gx1, region_y1, gx2, region_y2)

        best_plate = None
        best_score = 0.0
        for (px1, py1, px2, py2, pconf) in last_plate_boxes_global:
            pcx = (px1 + px2) / 2.0
            pcy = (py1 + py2) / 2.0
            if not (gx1 <= pcx <= gx2 and region_y1 <= pcy <= region_y2):
                continue

            score = iou_xyxy((px1, py1, px2, py2), region) * 0.7 + pconf * 0.3
            if score > best_score:
                best_score = score
                best_plate = (px1, py1, px2, py2)

        if best_plate is None or best_score < IOU_MIN_PLATE_TO_REGION:
            continue

        px1, py1, px2, py2 = best_plate
        plate_crop = frame[py1:py2, px1:px2]
        if plate_crop.size == 0:
            continue

        try:
            ocr_result = ocr.ocr(plate_crop)
            if ocr_result and ocr_result[0]:
                text = ocr_result[0][0][1][0].replace(" ", "").upper()
                if plate_pattern.match(text):
                    plate_history[tid].append(text)
                    if len(plate_history[tid]) >= CONSENSUS_FRAMES:
                        most_common = Counter(plate_history[tid]).most_common(1)[0][0]
                        plate_cache[tid] = {
                            "text": most_common,
                            "crop": plate_crop.copy(),  # ✅ bir kere kaydet, bir daha değişmez
                            "last_frame": frame_idx
                        }
        except:
            pass

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(frame)
    cv2.imshow("STABLE ANPR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ FINISHED")