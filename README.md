
# SmartANPR: Vehicle Access Management with YOLOv8 & EasyOCR

This project detects vehicles in a video, reads their license plates, and allows integration with a web app to manage access (e.g., for a gate system). The system uses **YOLOv8** for vehicle and license plate detection and **EasyOCR** for reading plate numbers.

---

## Features

1. Detect vehicles (car, bus, truck) from videos.
2. Detect and read license plates.
3. Track vehicles across frames using **SORT** tracker.
4. Export results to `test.csv`.
5. Whitelist functionality to allow only authorized vehicles.
6. Simple **Flask** web app to visualize entries and whitelist vehicles.

---

## Requirements

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- EasyOCR
- Flask
- NumPy
- SORT tracker (included in the repo)

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## Project Structure

```
automatic-number-plate-recognition-python-yolov8-main/
│
├─ data/
│   └─ video.mp4           # Input video file
│
├─ models/
│   ├─ yolov8n.pt          # Pretrained YOLOv8 model (COCO)
│   └─ best.pt             # License plate detector model
│
├─ sort/
│   └─ sort.py             # SORT tracker implementation
│
├─ util.py                 # Utility functions (get_car, read_license_plate, write_csv)
├─ main.py                 # Main script to run detection & tracking
├─ test.csv                # Output CSV from ANPR
├─ whitelist.csv           # Authorized vehicles
│
├─ web_app/
│   ├─ app.py              # Flask web app
│   └─ templates/
│       └─ index.html      # Web interface template
│
└─ README.md
```

---

## Usage

### Run ANPR on a video

```bash
# Activate virtual environment
./venv/Scripts/activate

# Run detection
python main.py --source data/video.mp4
```

* Output will be saved in `test.csv`.
* Each row contains: `frame_nmr, car_id, car_bbox, license_plate_bbox, license_plate_bbox_score, license_number, license_number_score`.

### Run the Web App

```bash
cd web_app
pip install flask  # if not already installed
python app.py
```

* Open `http://127.0.0.1:5000` in a browser.
* Displays vehicles detected and allows you to check against `whitelist.csv`.

---

## Web App Integration

* `test.csv` → Stores all detected license plate numbers.
* `whitelist.csv` → Stores authorized license plate numbers.
* Flask app reads both CSVs and shows which vehicles are allowed.
* Can be extended for a **gate system**:

  * Detect vehicle in real-time.
  * Check license plate against whitelist.
  * Open gate if authorized.

---

## Extending the Project

1. **Real-Time Camera Feed**

   * Replace video input with live camera feed using OpenCV.
2. **Database Integration**

   * Save entries to MySQL/PostgreSQL for historical tracking.
3. **Notifications**

   * Send alerts for unauthorized vehicle attempts.
4. **Gate System**

   * Connect with Arduino/Raspberry Pi to physically control a gate.
5. **Dashboard**

   * Add charts or logs for daily vehicle entries.

---

## References

* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [EasyOCR](https://github.com/JaidedAI/EasyOCR)
* [SORT Tracker](https://github.com/abewley/sort)
* [Flask](https://flask.palletsprojects.com/)

---
