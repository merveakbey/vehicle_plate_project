import cv2
import time
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import defaultdict

# ------------------------
# MODELS
# ------------------------
vehicle_model = YOLO("/home/merve/vehicle_plate_project/models/yolo26s.pt")
plate_model   = YOLO("/home/merve/vehicle_plate_project/models/license_plate_detector.pt")
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

# ------------------------
# VEHICLE CLASSES
# ------------------------
names = vehicle_model.names
if isinstance(names, dict):
    name_list = [names[i] for i in sorted(names.keys())]
else:
    name_list = list(names)

wanted = {"car", "truck", "bus", "motorcycle", "motorbike", "bicycle", "van"}
TARGET_CLASSES = [i for i, n in enumerate(name_list) if n.lower() in wanted]
if len(TARGET_CLASSES) == 0:
    TARGET_CLASSES = [3, 4, 5, 8]
print("✅ TARGET_CLASSES:", TARGET_CLASSES)

# ------------------------
# UK PLATE FILTER
# ------------------------
# UK standard: AA00AAA (ör: GX15OCJ)
uk_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

def uk_fix(text: str) -> str:
    """
    OCR çıktılarını UK formatına yaklaştırmak için küçük düzeltmeler.
    - 3. ve 4. karakter rakam olmalı => I/L->1, S->5, O->0
    - Harf olması gereken yerlerde 0/1 varsa => 0->O, 1->I
    """
    t = text.replace(" ", "").upper()
    if len(t) != 7:
        return t

    t = list(t)

    # digits positions: 2,3
    map_to_digit = {'I': '1', 'L': '1', 'S': '5', 'O': '0'}
    t[2] = map_to_digit.get(t[2], t[2])
    t[3] = map_to_digit.get(t[3], t[3])

    # letter positions: 0,1,4,5,6
    map_to_letter = {'0': 'O', '1': 'I'}
    for idx in [0, 1, 4, 5, 6]:
        t[idx] = map_to_letter.get(t[idx], t[idx])

    return "".join(t)

# ------------------------
# VIDEO
# ------------------------
VIDEO_PATH  = "/home/merve/vehicle_plate_project/License Plate Detection Test.mp4"
OUTPUT_PATH = "stable_anpr_output_UK_CLAHE_pipeline.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

# ------------------------
# ROI (center 640x640)
# ------------------------
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

# ------------------------
# PIPELINE: gray -> up3x -> bilateral -> CLAHE
# ------------------------
UPSCALE = 3

def preprocess_gray_bilat_clahe(plate_bgr):
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    den = cv2.bilateralFilter(up, d=7, sigmaColor=60, sigmaSpace=60)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(den)
    return clahe  # grayscale

def make_panel(gray_img, w=220, h=70):
    bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

# ------------------------
# STABILIZATION
# ------------------------
plate_cache = {}                   # tid -> {"text","panel","last_frame"}
plate_history = defaultdict(list)  # tid -> [(text, score), ...]
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

print("🚀 STABLE ANPR STARTED (UK FILTER + CLAHE pipeline)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if time.time() - start_video_time >= MAX_SECONDS:
        break

    frame_idx += 1
    start_time = time.time()

    # ROI draw
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    draw_corner_box(frame, x_min, y_min, x_max, y_max, color=(0,0,255), thickness=3, length=35)

    roi_crop = frame[y_min:y_max, x_min:x_max]

    # Vehicle tracking
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

    # cleanup old tracks
    dead_ids = [tid for tid, f in last_seen.items() if frame_idx - f > MAX_MISSED]
    for tid in dead_ids:
        last_seen.pop(tid, None)
        plate_cache.pop(tid, None)
        plate_history.pop(tid, None)

    # Plate detect in ROI
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

    # No IDs
    if results.boxes.id is None:
        fps = 1 / (time.time() - start_time + 1e-9)
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

        # ------------------------
        # CACHE: if exists, render panel
        # ------------------------
        if tid in plate_cache:
            data = plate_cache[tid]
            text = data["text"]
            panel_img = data["panel"]  # 220x70

            ph, pw = panel_img.shape[:2]
            panel_x = gx1
            panel_y = gy1 - ph - 60
            if panel_y < 0:
                panel_y = gy2 + 10

            panel_x = max(0, min(panel_x, width - pw))
            panel_y = max(0, min(panel_y, height - ph))

            frame[panel_y:panel_y+ph, panel_x:panel_x+pw] = panel_img
            cv2.rectangle(frame, (panel_x, panel_y-40), (panel_x+pw, panel_y), (255,255,255), -1)
            cv2.putText(frame, text, (panel_x+8, panel_y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            continue

        # ------------------------
        # Match plate box to this vehicle
        # ------------------------
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

        # ------------------------
        # PREPROCESS + OCR
        # ------------------------
        proc_gray = preprocess_gray_bilat_clahe(plate_crop)
        ocr_input = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2BGR)
        panel_img = make_panel(proc_gray, 220, 70)

        try:
            ocr_result = ocr.ocr(ocr_input)
            if ocr_result and ocr_result[0]:
                # En yüksek skorlu satırı seç
                best_txt, best_sc = None, -1.0
                for line in ocr_result[0]:
                    txt = line[1][0].replace(" ", "").upper()
                    sc  = float(line[1][1])
                    if sc > best_sc:
                        best_txt, best_sc = txt, sc

                if best_txt:
                    fixed = uk_fix(best_txt)

                    # ✅ sadece UK formatı kabul
                    if uk_pattern.match(fixed):
                        plate_history[tid].append((fixed, best_sc))

                        # 3 okuma sonra skor toplamı en yüksek olanı seç
                        if len(plate_history[tid]) >= CONSENSUS_FRAMES:
                            score_sum = {}
                            for t, s in plate_history[tid]:
                                score_sum[t] = score_sum.get(t, 0.0) + s

                            best_final = max(score_sum.items(), key=lambda x: x[1])[0]

                            plate_cache[tid] = {
                                "text": best_final,
                                "panel": panel_img.copy(),
                                "last_frame": frame_idx
                            }
        except:
            pass

    fps = 1 / (time.time() - start_time + 1e-9)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    out.write(frame)
    cv2.imshow("STABLE ANPR", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ FINISHED")
