# Real-Time YOLO Traffic Counter & Analytics Dashboard

An AI-powered real-time vehicle detection, tracking, and bi-directional counting system built with **YOLOv8**, **Roboflow Supervision**, **FastAPI**, and an interactive **Web Dashboard**.

This repository contains a suite of Python scripts and a modern Web Application dedicated to real-time vehicle detection, tracking, and counting, leveraging the **YOLO (You Only Look Once)** object detection model, specifically **YOLOv8**. Designed to facilitate vehicle counting across multi-lane highways and urban roads, this project integrates advanced computer vision tools like Roboflow Supervision for seamless annotation, object tracking, and detection smoothing, alongside live USB/IP camera integration and real-time Webhook notifications.

---

## 🧩 Core AI & Computer Vision Components

- **YOLOv8 Model**: Used for high-accuracy, real-time vehicle detection. YOLOv8 is particularly effective in distinguishing various vehicle types, including **cars, trucks, buses, and motorbikes**, which makes it highly suitable for traffic monitoring.
- **Roboflow Supervision Library**: Instrumental in annotating frames, visualizing bounding boxes, tracking objects, and implementing utilities such as line drawing and overlay creation. It supports **ByteTrack** for high-precision object tracking and **DetectionsSmoother** for enhanced tracking stability across video frames.
- **Flexible Configurations**: Tailored configurations offering different boundary conditions, multi-lane partitioning, live camera integration (USB & RTSP IP cameras), and custom Webhook alerts.

---

## 🌟 Key Features

### 📹 1. Flexible Video & Live Camera Inputs
- **USB / Integrated Webcams**: Stream directly from local camera indices (`0`, `1`, etc.).
- **IP / CCTV Cameras**: Connect directly to network RTSP streams (`rtsp://user:pass@ip:port/stream`) with automatic **FFMPEG + TCP transport** for zero connection timeouts.
- **Video File Uploads**: Upload pre-recorded traffic video files (`.mp4`, `.avi`, `.mov`) directly through the Web Dashboard.

### 📊 2. Real-Time Web Analytics Dashboard
- **Live Video Stream**: Low-latency MJPEG video feed overlaid with real-time detection bounding boxes, tracking lines, and count metrics.
- **Live Statistics Cards**: Real-time counter metrics tracking **Total Vehicles**, **Upward / Inbound Flow**, and **Downward / Outbound Flow**.
- **Dynamic Vehicle Filtering**: Selectively track and count specific vehicle classes in real-time:
  - 🚗 Cars
  - 🚚 Trucks
  - 🚌 Buses
  - 🏍️ Motorbikes

### 🔗 3. Webhook Event Dispatching
- Automatically send payload notifications whenever vehicles cross counting boundaries.
- Secure header authentication via custom `X-API-Key` token (`X-API-Key: my_secure_camera_token_123`).

---

## 📁 Repository Structure

```
.
├── app/
│   ├── main.py                     # FastAPI application & API endpoints
│   ├── services/
│   │   └── counter_service.py      # Core video processing, YOLO & ByteTrack engine
│   └── static/                     # Web Dashboard UI (HTML, CSS, JS)
├── DATA/
│   ├── INPUTS/                     # Uploaded video input folder
│   └── OUTPUTS/                    # Output logs & renders
├── yolo_car_counter_2.py           # Standalone OpenCV single-line counter
├── yolo_car_counter_3.py           # Standalone OpenCV elliptical tracker
├── yolo_car_counter_4.py           # Standalone multi-lane dual-direction counter
├── yolo_car_counter_5.py           # Standalone smooth overlay tracker
├── webhook_receiver.py             # Test tool to verify live webhook payloads
├── setup_and_run.bat               # One-click Windows setup & launcher script
├── requirements.txt                # Python dependencies
├── LICENSE                         # Repository license
└── README.md                       # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.8 or higher installed on your system.
- Webcam or RTSP Camera / Sample video file.

### 2. Installation

Clone the repository:
```bash
git clone https://github.com/Ladyunicorn560/Realtime-yolo-traffic-counter.git
cd Realtime-yolo-traffic-counter
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Running the Web Application

#### Option A: One-Click Windows Script
Simply double-click `setup_and_run.bat` or run:
```cmd
setup_and_run.bat
```

#### Option B: Direct Python Launch
```bash
python -m app.main
```

Once running, open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

## 📹 How to Connect Live Cameras

1. Open the Web Dashboard at `http://localhost:8000`.
2. In the **Camera Source** panel:
   - For **Local Webcam**: Enter `0` (or `1` for external USB webcam).
   - For **IP CCTV Camera**: Enter your camera's RTSP URL:
     ```text
     rtsp://admin:password@192.168.1.100:554/h264Preview_01_main
     ```
   - For **Video Upload**: Click **Upload Video File** and select your video.
3. Click **Start Stream**.

---

## 📡 Webhook Integration & Testing

To test live webhook payloads:

1. Launch the local webhook receiver in a separate terminal:
   ```bash
   python webhook_receiver.py
   ```
2. In the Web Dashboard, set the **Webhook URL** to:
   ```text
   http://127.0.0.1:9000/webhook
   ```
3. Whenever a vehicle crosses a line, structured JSON event payloads with class details, direction, count, and timestamp will be printed in the receiver terminal.

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the interactive Web Analytics Dashboard |
| `/video_feed` | `GET` | Live MJPEG video stream with YOLO annotations |
| `/stats` | `GET` | Returns current total, up, and down count stats (JSON) |
| `/start` | `POST` / `GET` | Starts camera or video stream (`{"source": "0"}`) |
| `/stop` | `POST` | Stops the running video stream |
| `/upload_video` | `POST` | Uploads a local video file and launches detection |

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
