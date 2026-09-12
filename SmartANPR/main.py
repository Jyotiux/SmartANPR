from ultralytics import YOLO
import cv2
import numpy as np

import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv, aggregate_plates_by_car, write_aggregated_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO("yolov8n.pt")  # vehicle detector
license_plate_detector = YOLO("./models/best.pt")  # your trained LP detector

# load video
cap = cv2.VideoCapture("./data/sample.mp4")
if not cap.isOpened():
    raise Exception("❌ Could not open input video!")

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input video: {width}x{height} @ {fps} FPS")

# Try safe codec for mp4, if fails fallback to avi
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
if not out.isOpened():
    print("⚠️ mp4v codec failed, switching to MJPG (.avi instead)")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

vehicles = [2, 3, 5, 7]

frame_nmr = -1
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    results[frame_nmr] = {}

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # detect license plates
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id != -1:
            # crop license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

            if license_plate_crop.size == 0:
                continue  # skip empty crops

            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            # OCR
            license_plate_text, license_plate_text_score = read_license_plate(
                license_plate_crop_thresh
            )

            if license_plate_text:
                results[frame_nmr][car_id] = {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox": [x1, y1, x2, y2],
                        "text": license_plate_text,
                        "bbox_score": score,
                        "text_score": license_plate_text_score,
                    },
                }

                # draw box and text
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    license_plate_text,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    # write frame
    out.write(frame)

# cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

write_csv(results, "./test.csv")

# Vehicle-level aggregation: pick one final plate per car_id using frequency +
# confidence across frames. Written to a separate file so test.csv is unchanged.
aggregated = aggregate_plates_by_car(results)
write_aggregated_csv(aggregated, "./test_aggregated.csv")

print("✅ Done! Video saved as output.mp4 (or output.avi).")
print("   Per-frame results in test.csv; aggregated per-vehicle plates in test_aggregated.csv")
