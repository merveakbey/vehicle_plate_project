import cv2
import numpy as np
import re
import time
import csv
from collections import defaultdict, Counter
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ==================================
# PATHS
# ==================================
VIDEO_PATH = "/home/merve/vehicle_plate_project/License Plate Detection Test.mp4"
OUTPUT_PATH = "result_video_production.mp4"
CSV_PATH = "plates_log.csv"

# ==================================
# MODELLER
# ==================================
vehicle_model = YOLO("/home/merve/vehicle_plate_project/models/yolo26nvisdroneboat.pt")
plate_model = YOLO("/home/merve/vehicle_plate_project/models/license_plate_detector.pt")

ocr = PaddleOCR(
    lang='en',
    use_gpu=False,
    show_log=False,
    use_angle_cls=True
)

# ==================================
# AYARLAR
# ==================================
VEHICLE_CLASSES = [3, 4, 5, 8]
VEHICLE_CONF = 0.30
PLATE_CONF = 0.30
MIN_VEHICLE_HEIGHT = 80
MIN_PLATE_WIDTH = 60
FRAME_SKIP = 2
LOCK_AFTER_VOTES = 6

# ==================================
# TRACKING
# ==================================
track_id = 0
tracks = {}
plate_votes = defaultdict(list)
locked_plates = {}
saved_plates = set()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union != 0 else 0

# ==================================
# CHARACTER FUSION
# ==================================
def character_voting(plate_list):
    plate_list = [p for p in plate_list if p != "UNKNOWN"]
    if not plate_list:
        return "UNKNOWN"

    max_len = max(len(p) for p in plate_list)
    final_plate = ""

    for i in range(max_len):
        chars = []
        for plate in plate_list:
            if len(plate) > i:
                chars.append(plate[i])
        if chars:
            final_plate += Counter(chars).most_common(1)[0][0]

    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$'
    if re.match(pattern, final_plate):
        return final_plate

    return final_plate

# ==================================
# CLEAN OCR
# ==================================
def clean_plate(text):
    text = text.upper().replace(" ", "")
    text = re.sub(r'[^A-Z0-9]', '', text)

    replacements = {
        "O":"0", "I":"1", "S":"5",
        "B":"8", "G":"6"
    }

    text = list(text)
    for i in range(len(text)):
        if text[i] in replacements:
            text[i] = replacements[text[i]]
    return "".join(text)

# ==================================
# VIDEO INIT
# ==================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

prev_time = time.time()
frame_count = 0

# CSV başlık
with open(CSV_PATH, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Plate"])

print("🚀 PRODUCTION ANPR STARTED")

# ==================================
# LOOP
# ==================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original = frame.copy()
    current_ids = set()

    vehicle_results = vehicle_model(frame, verbose=False)[0]

    for box in vehicle_results.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls not in VEHICLE_CLASSES or conf < VEHICLE_CONF:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (y2 - y1) < MIN_VEHICLE_HEIGHT:
            continue

        vehicle_box = [x1, y1, x2, y2]

        matched_id = None
        for tid, tbox in tracks.items():
            if iou(vehicle_box, tbox) > 0.4:
                matched_id = tid
                break

        if matched_id is None:
            track_id += 1
            matched_id = track_id

        tracks[matched_id] = vehicle_box
        current_ids.add(matched_id)

        cv2.rectangle(original, (x1,y1), (x2,y2), (0,255,0), 2)

        if frame_count % FRAME_SKIP != 0:
            continue

        vehicle_crop = frame[y1:y2, x1:x2]
        plate_results = plate_model(vehicle_crop, verbose=False)[0]

        for pbox in plate_results.boxes:

            if float(pbox.conf[0]) < PLATE_CONF:
                continue

            px1, py1, px2, py2 = map(int, pbox.xyxy[0])

            if (px2 - px1) < MIN_PLATE_WIDTH:
                continue

            pad = 20
            px1 = max(0, px1 - pad)
            py1 = max(0, py1 - pad)
            px2 = min(vehicle_crop.shape[1], px2 + pad)
            py2 = min(vehicle_crop.shape[0], py2 + pad)

            plate_crop = vehicle_crop[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            # OCR improve
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0, (8,8))
            gray = clahe.apply(gray)
            gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            plate_text = "UNKNOWN"
            try:
                result = ocr.ocr(thresh)
                if result and result[0]:
                    plate_text = result[0][0][1][0]
            except:
                pass

            plate_text = clean_plate(plate_text)
            plate_votes[matched_id].append(plate_text)

            if matched_id not in locked_plates and \
               len(plate_votes[matched_id]) >= LOCK_AFTER_VOTES:

                fused = character_voting(plate_votes[matched_id])
                locked_plates[matched_id] = fused
                print("🔒 LOCKED:", fused)

            if matched_id in locked_plates:
                display_plate = locked_plates[matched_id]

                gx1 = x1 + px1
                gy1 = y1 + py1
                gx2 = x1 + px2
                gy2 = y1 + py2

                cv2.rectangle(original, (gx1,gy1), (gx2,gy2), (0,0,255), 2)
                cv2.putText(original, display_plate,
                            (gx1, gy1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

    # FPS hesaplama
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(original,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2)

    cv2.imshow("PRODUCTION ANPR", original)
    out.write(original)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ FINISHED:", OUTPUT_PATH)
print("📄 CSV:", CSV_PATH)