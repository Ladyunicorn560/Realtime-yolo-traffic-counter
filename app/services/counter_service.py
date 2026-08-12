import cv2 as cv
import numpy as np
from ultralytics import YOLO
import supervision as sv
import threading
import time
import os
import requests
from datetime import datetime, timezone


def _did_cross(prev_y: int, curr_y: int, line_y: int) -> bool:
    """
    True if the segment from prev_y to curr_y crosses line_y.
    This is the standard approach used in professional traffic systems —
    a vehicle moving at any speed will be caught in a single frame transition.
    """
    return (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y)


class CounterService:
    def __init__(self, model_path="yolov8m.pt"):
        self.model = YOLO(model_path)

        try:
            self.tracker = sv.ByteTrack(
                track_thresh=0.25,
                track_buffer=120,
                match_thresh=0.8,
                frame_rate=30
            )
        except TypeError:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=120,
                minimum_matching_threshold=0.8,
                frame_rate=30
            )

        self.class_names = self.model.names

        # FIX: Corrected COCO class IDs.
        # Old code had class 1 (bicycle) mapped as 'motorbike' — wrong!
        # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = {
            2: 'car',
            3: 'motorbike',
            5: 'bus',
            7: 'truck'
        }

        self.selected_classes = list(self.vehicle_classes.keys())

        self.running = False
        self.cap = None
        self.source = None

        self.latest_raw_frame = None
        self.current_frame = None

        # FIX: Thread-safety locks.
        # _capture_loop writes latest_raw_frame; _process_loop reads it → race condition.
        # counts are written in _process_loop and read by /stats API → another race condition.
        self._frame_lock = threading.Lock()
        self._counts_lock = threading.Lock()

        self.counts = {
            "total": 0,
            "up": 0,
            "down": 0,
            "car": 0,
            "motorbike": 0,
            "bus": 0,
            "truck": 0
        }

        self.crossed_ids = set()
        # FIX: Store only the last Y position per track — all we need for
        # single-frame line-segment crossing detection.
        self.last_positions = {}

        # First-position tracking — prevents double-counting when ByteTrack
        # drops and re-assigns a new track_id to the same physical vehicle.
        # A track is only allowed to count if it crosses FROM the side it
        # first appeared on — so a re-assigned track that starts below the
        # line cannot be counted going DOWN again.
        self.track_first_side = {}      # track_id -> "above" | "below"
        self.track_last_cross_time = {} # track_id -> timestamp (prevents rapid line jitter)

        self.line_fraction = 0.65
        self.limits = [0, 480, 1280, 480]

        # Webhook — empty string means disabled
        self.webhook_url = ""
        self.webhook_api_key = "my_secure_camera_token_123"

        # JPEG cache — encoded once per frame in _process_loop,
        # served directly by get_frame() to avoid repeated CPU-heavy encoding
        self.current_jpeg = None
        self._jpeg_lock   = threading.Lock()

        self.box_annotator = sv.RoundBoxAnnotator(
            color_lookup=sv.ColorLookup.TRACK
        )

        self.label_annotator = sv.LabelAnnotator(
            text_position=sv.Position.TOP_CENTER,
            color_lookup=sv.ColorLookup.TRACK
        )

        self.trace_annotator = sv.TraceAnnotator(
            color_lookup=sv.ColorLookup.TRACK
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start_stream(self, source=0):
        print(f"[CounterService] Starting stream: {source}")

        if self.running:
            print("Stream already running")
            return

        with self._counts_lock:
            self.counts = {
                "total": 0,
                "up": 0,
                "down": 0,
                "car": 0,
                "motorbike": 0,
                "bus": 0,
                "truck": 0
            }

        self.last_positions = {}
        self.track_first_side = {}
        self.track_last_cross_time = {}

        if str(source).isdigit():
            source = int(source)

        self.source = source
        self.cap = self._open_capture(source)

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open stream: {source}")
            return

        print("✅ Camera connected")
        self.running = True

        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.capture_thread.start()

        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self.process_thread.start()

    def stop_stream(self):
        self.running = False

        if self.cap:
            self.cap.release()

        self.cap = None

        with self._frame_lock:
            self.latest_raw_frame = None

        self.last_positions = {}

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_capture(self, source):
        """Open VideoCapture with FFMPEG+TCP — same as working ALPR project.
        Plain cv.VideoCapture uses UDP for RTSP which times out on this camera.
        """
        if isinstance(source, int):
            return cv.VideoCapture(source)
        # Force TCP transport — prevents the 30s UDP timeout
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;recv_buffer_size;10485760"
        return cv.VideoCapture(source, cv.CAP_FFMPEG)

    def _capture_loop(self):
        while self.running and self.cap.isOpened():
            success, frame = self.cap.read()

            if success:
                # FIX: Lock before writing shared frame — prevents race condition
                # with _process_loop reading on another thread.
                with self._frame_lock:
                    self.latest_raw_frame = frame
            else:
                print("⚠️ Frame dropped. Reconnecting...")
                self.cap.release()
                time.sleep(1)
                self.cap = self._open_capture(self.source)

                if not self.cap.isOpened():
                    print("❌ Reconnect failed")
                    time.sleep(1)
                    continue

                print("✅ Reconnected")
                continue

            time.sleep(0.001)

    def _process_loop(self):
        frame_idx    = 0
        stride       = 3          # Run YOLO every 3rd frame — frees more CPU for display
        target_width = 800
        last_detections = None    # persist last YOLO result for non-YOLO frames

        while self.running:
            # FIX: Lock before reading shared frame
            with self._frame_lock:
                raw = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

            if raw is None:
                time.sleep(0.01)
                continue

            frame = raw
            frame_idx += 1

            if frame_idx % stride == 0:
                h_orig, w_orig = frame.shape[:2]
                scale = target_width / w_orig

                roi_resized = cv.resize(
                    frame,
                    (target_width, int(h_orig * scale))
                )

                y_offset = int(120 * scale)
                mask = np.zeros_like(roi_resized, dtype=np.uint8)
                mask[y_offset:, :] = 255
                ROI = cv.bitwise_and(roi_resized, mask)

                results = self.model(
                    ROI,
                    verbose=False,
                    imgsz=800,
                    conf=0.20,      # Lowered from 0.35 — catches close-up/angled vehicles
                    iou=0.45,
                    agnostic_nms=True
                )[0]

                detections = sv.Detections.from_ultralytics(results)
                detections.xyxy = detections.xyxy / scale
                detections = self.tracker.update_with_detections(detections)

                if detections.tracker_id is not None:
                    class_mask = np.isin(detections.class_id, self.selected_classes)
                    detections = detections[class_mask]

                    # FIX: Counting line expressed as fraction of frame height —
                    # works correctly regardless of stream resolution.
                    line_y = int(frame.shape[0] * self.line_fraction)
                    self.limits = [0, line_y, frame.shape[1], line_y]

                    for track_id, class_id, center_point in zip(
                        detections.tracker_id,
                        detections.class_id,
                        detections.get_anchors_coordinates(
                            anchor=sv.Position.BOTTOM_CENTER
                        )
                    ):
                        cx, cy = map(int, center_point)

                        # Record which side of the line this track first appeared on.
                        # Used below to prevent false double-counts.
                        if track_id not in self.track_first_side:
                            self.track_first_side[track_id] = "above" if cy < line_y else "below"

                        prev_y = self.last_positions.get(track_id)
                        self.last_positions[track_id] = cy   # always update

                        # FIX: Proper line-segment crossing detection.
                        # Old code check for crossed_ids blocked reversals. We now use
                        # track_last_cross_time (2s jitter cooldown per track) and toggle
                        # track_first_side to allow multiple alternating direction crossings.
                        if (prev_y is not None
                                and _did_cross(prev_y, cy, line_y)):

                            # Prevent rapid double-triggering on the line due to jitter
                            now = time.time()
                            if now - self.track_last_cross_time.get(track_id, 0.0) < 2.0:
                                continue

                            # FIX: Direction determined AT crossing moment using
                            # prev_y (frame before) vs cy (frame after crossing).
                            direction = "up" if cy < prev_y else "down"
                            type_name = self.vehicle_classes.get(class_id)

                            # First-position guard:
                            # Only count if the vehicle started on the side it
                            # is crossing FROM.
                            first_side = self.track_first_side.get(track_id, "above")
                            if direction == "down" and first_side != "above":
                                continue   # started below — skip
                            if direction == "up" and first_side != "below":
                                continue   # started above — skip

                            # Update side and set last cross timestamp so it can reverse later
                            self.track_first_side[track_id] = "below" if direction == "down" else "above"
                            self.track_last_cross_time[track_id] = now

                            # FIX: Lock counts before writing — /stats reads on another thread
                            with self._counts_lock:
                                self.counts["total"] += 1
                                self.counts[direction] += 1
                                if type_name in self.counts:
                                    self.counts[type_name] += 1
                                snapshot = dict(self.counts)

                            # Fire webhook in background (non-blocking)
                            if self.webhook_url:
                                payload = {
                                    "camera_id": str(self.source) if self.source is not None else "camera_1",
                                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "plate": {
                                        "number": f"{(type_name or 'vehicle').upper()}-{int(track_id)}"
                                    },
                                    "confidence": 1.0,
                                    "vehicle": {
                                        "type": type_name or "unknown",
                                        "track_id": int(track_id),
                                        "direction": direction
                                    },
                                    "counts": snapshot
                                }
                                def _send_webhook(url=self.webhook_url, key=self.webhook_api_key, p=payload):
                                    try:
                                        import json
                                        headers = {"X-API-Key": key, "Content-Type": "application/json"}
                                        print("\n" + "="*50)
                                        print("🚀 📤 SENDING WEBHOOK TO REMOTE SERVER:")
                                        print(f"🔗 URL: {url}")
                                        print(f"📦 Payload: {json.dumps(p, indent=2)}")
                                        print("="*50 + "\n", flush=True)
                                        response = requests.post(url, json=p, headers=headers, timeout=3)
                                        print(f"📡 Webhook response | Status: {response.status_code} | Body: {response.text}", flush=True)
                                    except Exception as exc:
                                        print(f"❌ Webhook failed → {url} | Error: {exc}", flush=True)
                                threading.Thread(target=_send_webhook, daemon=True).start()

                    last_detections = detections   # persist for non-YOLO frames

                    frame = self.box_annotator.annotate(
                        frame, detections=detections
                    )

                    labels = [
                        f"{self.vehicle_classes.get(cid, 'v')} #{tid}"
                        for tid, cid in zip(
                            detections.tracker_id,
                            detections.class_id
                        )
                    ]

                    frame = self.label_annotator.annotate(
                        frame, detections=detections, labels=labels
                    )

            else:
                # Non-YOLO frame: re-draw last known detections so boxes stay
                # visible continuously instead of flickering on every 3rd frame
                if last_detections is not None and len(last_detections) > 0:
                    frame = self.box_annotator.annotate(frame, detections=last_detections)
                    labels = [
                        f"{self.vehicle_classes.get(cid, 'v')} #{tid}"
                        for tid, cid in zip(
                            last_detections.tracker_id,
                            last_detections.class_id
                        )
                    ]
                    frame = self.label_annotator.annotate(
                        frame, detections=last_detections, labels=labels
                    )

            # Draw counting line on every frame (not just processed ones)
            line_y = int(frame.shape[0] * self.line_fraction)
            cv.line(
                frame,
                (0, line_y),
                (frame.shape[1], line_y),
                (0, 255, 0),
                2
            )

            self.current_frame = frame

            # Encode JPEG once here and cache — eliminates repeated encoding
            # on every HTTP /frame request (critical when multiple browser tabs open)
            ret, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                with self._jpeg_lock:
                    self.current_jpeg = buf.tobytes()

            time.sleep(0.003)   # ~333 fps cap — tighter loop for smoother display

    def get_frame(self):
        with self._jpeg_lock:
            return self.current_jpeg   # Return pre-encoded bytes — zero CPU cost

    def get_counts(self):
        # FIX: Return a snapshot copy under lock — prevents torn reads
        # when /stats and _process_loop run concurrently.
        with self._counts_lock:
            return dict(self.counts)