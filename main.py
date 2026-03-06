import cv2
import time
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ------------------------
# MODELS
# ------------------------
vehicle_model = YOLO("/home/merve/vehicle_plate_project/models/yolo26nvisdroneboat.pt")
plate_model = YOLO("/home/merve/vehicle_plate_project/models/license_plate_detector.pt")
ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

TARGET_CLASSES = [3, 4, 5, 8]
plate_pattern = re.compile(r'^[A-Z0-9]{5,8}$')

VIDEO_PATH = "/home/merve/vehicle_plate_project/License Plate Detection Test.mp4"
OUTPUT_PATH = "traffic_panel_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_video = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

# ROI
x_min = int(0.20 * width)
x_max = int(0.98 * width)
y_min = int(0.38 * height)
y_max = height

plate_cache = {}

def draw_corner_box(img, x1, y1, x2, y2, color=(0,255,0), thickness=3, length=25):
    cv2.line(img, (x1,y1), (x1+length,y1), color, thickness)
    cv2.line(img, (x1,y1), (x1,y1+length), color, thickness)
    cv2.line(img, (x2,y1), (x2-length,y1), color, thickness)
    cv2.line(img, (x2,y1), (x2,y1+length), color, thickness)
    cv2.line(img, (x1,y2), (x1+length,y2), color, thickness)
    cv2.line(img, (x1,y2), (x1,y2-length), color, thickness)
    cv2.line(img, (x2,y2), (x2-length,y2), color, thickness)
    cv2.line(img, (x2,y2), (x2,y2-length), color, thickness)

print("🚀 PANEL ANPR STARTED")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    roi_crop = frame[y_min:y_max, x_min:x_max]

    results = vehicle_model.track(roi_crop, persist=True, verbose=False)[0]

    if results.boxes.id is not None:

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        ids = results.boxes.id.cpu().numpy()

        for box, cls, track_id in zip(boxes, classes, ids):

            if int(cls) not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box)

            gx1 = x1 + x_min
            gy1 = y1 + y_min
            gx2 = x2 + x_min
            gy2 = y2 + y_min

            vehicle_crop = frame[gy1:gy2, gx1:gx2]
            if vehicle_crop.size == 0:
                continue

            if (gx2 - gx1) * (gy2 - gy1) < 25000:
                continue

            draw_corner_box(frame, gx1, gy1, gx2, gy2)

            # OCR only once
            if track_id not in plate_cache:

                plate_results = plate_model(vehicle_crop, verbose=False)[0]

                for pbox in plate_results.boxes:

                    if float(pbox.conf[0]) < 0.6:
                        continue

                    px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                    plate_crop = vehicle_crop[py1:py2, px1:px2]

                    if plate_crop.size == 0:
                        continue

                    try:
                        ocr_result = ocr.ocr(plate_crop)

                        if ocr_result and ocr_result[0]:
                            text = ocr_result[0][0][1][0]
                            text = text.replace(" ", "").upper()

                            if plate_pattern.match(text):

                                plate_cache[track_id] = {
                                    "text": text,
                                    "crop": plate_crop.copy()
                                }
                                break
                    except:
                        pass

            if track_id in plate_cache:

                data = plate_cache[track_id]
                text = data["text"]
                plate_crop = data["crop"]

                # Resize crop
                crop_w = 220
                crop_h = 70
                plate_resized = cv2.resize(plate_crop, (crop_w, crop_h))

                # Panel position (üstte)
                panel_x = gx1
                panel_y = gy1 - crop_h - 60

                if panel_y < 0:
                    panel_y = gy2 + 10

                # Crop
                frame[panel_y:panel_y+crop_h,
                      panel_x:panel_x+crop_w] = plate_resized

                # White text panel
                cv2.rectangle(frame,
                              (panel_x, panel_y-40),
                              (panel_x+crop_w, panel_y),
                              (255,255,255),
                              -1)

                cv2.putText(frame,
                            text,
                            (panel_x+20, panel_y-12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0,0,0),
                            2)

    fps = 1 / (time.time() - start_time)
    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    out.write(frame)
    cv2.imshow("PANEL ANPR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ FINISHED")