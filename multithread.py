import cv2
import time
import numpy as np
import re
from ultralytics import YOLO
from collections import defaultdict, Counter, deque
import multiprocessing as mp

# ------------------------
# CONFIG
# ------------------------
OCR_PROCESSES = 2
MAX_INFLIGHT = 4

CONSENSUS_FRAMES = 2
PLATE_EVERY_N = 2           # 2 iyi; istersen 1
PLATE_CONF_MIN = 0.45       # 0.60 çok katı olabiliyor
IOU_MIN_PLATE_TO_REGION = 0.06

MAX_MISSED = 25
MAX_SECONDS = 130

ROI_SIZE = 640
TRACK_IMGSZ = 512

# Kilitlenmeden önce panelde crop gösterilsin mi?
SHOW_PREVIEW_BEFORE_LOCK = True   # istemezsen False yap
PREVIEW_TEXT = "..."              # kilitlenmeden önce yazı

# UK pattern: AA11AAA (boşluk yok)
uk_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')

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

def prep_for_ocr(crop_bgr):
    # küçük plaka için büyüt + kontrast
    crop = cv2.resize(crop_bgr, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def normalize_uk(text):
    """
    OCR çıktısını UK formatına zorla yaklaştır:
    - sadece A-Z0-9 bırak
    - uzunluk 7 değilse None
    - pozisyonlara göre 0/O, 1/I, 5/S, 8/B gibi karışmaları düzelt
      AA 11 AAA
      0-1 letter, 2-3 digit, 4-6 letter
    """
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(t) != 7:
        return None

    t = list(t)

    # Harf beklenen yerler
    letter_pos = [0, 1, 4, 5, 6]
    # Rakam beklenen yerler
    digit_pos = [2, 3]

    # Digit pozisyonlarında harfleri rakama yaklaştır
    for i in digit_pos:
        if t[i] == 'O': t[i] = '0'
        if t[i] == 'I': t[i] = '1'
        if t[i] == 'Z': t[i] = '2'
        if t[i] == 'S': t[i] = '5'
        if t[i] == 'B': t[i] = '8'

    # Letter pozisyonlarında rakamları harfe yaklaştır
    for i in letter_pos:
        if t[i] == '0': t[i] = 'O'
        if t[i] == '1': t[i] = 'I'
        if t[i] == '2': t[i] = 'Z'
        if t[i] == '5': t[i] = 'S'
        if t[i] == '8': t[i] = 'B'

    out = ''.join(t)
    return out if uk_pattern.match(out) else None

# ------------------------
# OCR WORKER (Process)
# ------------------------
def ocr_worker(in_q: mp.Queue, out_q: mp.Queue):
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

    while True:
        item = in_q.get()
        if item is None:
            break

        tid, crop_bgr = item
        text_out = None

        try:
            crop = prep_for_ocr(crop_bgr)
            res = ocr.ocr(crop)
            if res and res[0]:
                raw = res[0][0][1][0]
                norm = normalize_uk(raw)
                if norm is not None:
                    text_out = norm
        except:
            text_out = None

        out_q.put((tid, text_out))

def main():
    vehicle_model = YOLO("/home/merve/vehicle_plate_project/models/yolo26s.pt")
    plate_model   = YOLO("/home/merve/vehicle_plate_project/models/license_plate_detector.pt")

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

    VIDEO_PATH  = "/home/merve/vehicle_plate_project/License Plate Detection Test.mp4"
    OUTPUT_PATH = "stable_anpr_output.mp4"

    cap = cv2.VideoCapture(VIDEO_PATH)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video, (width, height))

    # ROI
    roi_w = min(ROI_SIZE, width)
    roi_h = min(ROI_SIZE, height)
    x_min = max(0, (width - roi_w) // 2)
    y_min = max(0, (height - roi_h) // 2)
    x_max = x_min + roi_w
    y_max = y_min + roi_h

    # IPC
    in_q  = mp.Queue(maxsize=MAX_INFLIGHT + 6)
    out_q = mp.Queue()

    workers = []
    for _ in range(OCR_PROCESSES):
        p = mp.Process(target=ocr_worker, args=(in_q, out_q), daemon=True)
        p.start()
        workers.append(p)

    pending = set()
    pending_queue = deque()

    # caches
    plate_cache = {}                   # tid -> {"text": str|None, "crop": bgr, "last_frame": int}
    plate_history = defaultdict(list)
    last_seen = {}
    frame_idx = 0
    last_plate_boxes_global = []

    start_video_time = time.time()
    print("🚀 STABLE ANPR STARTED")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if time.time() - start_video_time >= MAX_SECONDS:
            break

        frame_idx += 1
        start_time = time.time()

        # 1) OCR sonuçlarını topla
        while True:
            try:
                tid, text = out_q.get_nowait()
            except:
                break

            pending.discard(tid)
            if tid in plate_cache and plate_cache[tid].get("text") is None and text is not None:
                plate_history[tid].append(text)
                if len(plate_history[tid]) >= CONSENSUS_FRAMES:
                    most_common = Counter(plate_history[tid]).most_common(1)[0][0]
                    plate_cache[tid]["text"] = most_common

        # 2) OCR işleri gönder
        while pending_queue and len(pending) < MAX_INFLIGHT and not in_q.full():
            tid, crop = pending_queue.popleft()
            if tid in plate_cache and plate_cache[tid].get("text") is None and tid not in pending:
                in_q.put((tid, crop), block=False)
                pending.add(tid)

        # ROI çiz
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,0,255), 2)

        roi_crop = frame[y_min:y_max, x_min:x_max]

        # Araç track
        results = vehicle_model.track(
            roi_crop,
            persist=True,
            verbose=False,
            imgsz=TRACK_IMGSZ,
            classes=TARGET_CLASSES,
            conf=0.35,
            iou=0.50,
            tracker="bytetrack.yaml"
        )[0]

        # expire
        dead_ids = [tid for tid, f in last_seen.items() if frame_idx - f > MAX_MISSED]
        for tid in dead_ids:
            last_seen.pop(tid, None)
            plate_cache.pop(tid, None)
            plate_history.pop(tid, None)
            pending.discard(tid)

        # Plaka tespiti (ROI) — her N frame
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

        if results.boxes.id is not None:
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

                # panel yerleri
                panel_x = gx1
                panel_y = gy1 - 70 - 60
                if panel_y < 0:
                    panel_y = gy2 + 10
                panel_x = max(0, min(panel_x, width - 220))
                panel_y = max(0, min(panel_y, height - 70))

                # kilitli ise çiz
                if tid in plate_cache and plate_cache[tid].get("text") is not None:
                    data = plate_cache[tid]
                    text = data["text"]
                    plate_resized = cv2.resize(data["crop"], (220, 70))
                    frame[panel_y:panel_y+70, panel_x:panel_x+220] = plate_resized
                    cv2.rectangle(frame, (panel_x, panel_y-40), (panel_x+220, panel_y), (255,255,255), -1)
                    cv2.putText(frame, text, (panel_x+20, panel_y-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
                    continue

                # kilitli değilse: plaka eşle
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

                    plate_w = (px2 - px1)
                    if not (0.12 * vehicle_w <= plate_w <= 0.65 * vehicle_w):
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

                # cache (crop sabit)
                if tid not in plate_cache:
                    plate_cache[tid] = {"text": None, "crop": plate_crop.copy(), "last_frame": frame_idx}
                else:
                    plate_cache[tid]["last_frame"] = frame_idx

                # ✅ kilitlenmeden önce preview göster (plaka tespiti var mı görürsün)
                if SHOW_PREVIEW_BEFORE_LOCK:
                    plate_resized = cv2.resize(plate_cache[tid]["crop"], (220, 70))
                    frame[panel_y:panel_y+70, panel_x:panel_x+220] = plate_resized
                    cv2.rectangle(frame, (panel_x, panel_y-40), (panel_x+220, panel_y), (255,255,255), -1)
                    cv2.putText(frame, PREVIEW_TEXT, (panel_x+20, panel_y-12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

                # OCR kuyruğuna ekle
                if plate_cache[tid]["text"] is None and tid not in pending:
                    if len(pending) + len(pending_queue) < MAX_INFLIGHT + 6:
                        pending_queue.append((tid, plate_crop.copy()))

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

    # workers stop
    for _ in workers:
        in_q.put(None)
    for p in workers:
        p.join(timeout=1)

    print("✅ FINISHED")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()