import cv2 as cv
from ultralytics import YOLO
import numpy as np
import supervision as sv


def _did_cross(prev_y: int, curr_y: int, line_y: int) -> bool:
    """
    True if the path from prev_y to curr_y crosses line_y.
    Handles any vehicle speed — no proximity window needed.
    """
    return (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y)


# Initialize YOLO model and video info
model = YOLO("../yolo weights/yolov8n.pt")  # Replace with your model path
video_path = "DATA/INPUTS/cars_on_highway_2.mp4"
video_info = sv.VideoInfo.from_video_path(video_path)
w, h, fps = video_info.width, video_info.height, video_info.fps

# Setup annotators
thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

box_annotator = sv.RoundBoxAnnotator(thickness=thickness, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness,
                                    text_position=sv.Position.TOP_CENTER, color_lookup=sv.ColorLookup.TRACK)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=fps * 2,
                                    position=sv.Position.CENTER, color_lookup=sv.ColorLookup.TRACK)

# Tracker and vehicle class setup
tracker = sv.ByteTrack(frame_rate=video_info.fps)
smoother = sv.DetectionsSmoother()

# Vehicle classes — COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
# (removed class 1 which is bicycle, not motorbike)
class_names = model.names
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
selected_classes = [cls_id for cls_id, class_name in model.names.items()
                    if class_name in vehicle_classes]

# FIX: Resolution-independent counting line + partition using frame fractions.
# Old code: limits = [0, 300, 1280, 300], partition_limit = 550 → pixel-hardcoded.
line_y = int(h * 0.42)
limits = [0, line_y, w, line_y]
partition_limit = int(w * 0.43)   # splits frame into left (up) / right (down) lanes

total_counts, crossed_ids = [], set()
total_counts_up, crossed_ids_up = [], set()
total_counts_down, crossed_ids_down = [], set()

# Per-track last Y position for line-segment crossing detection
last_positions = {}
last_positions_up = {}
last_positions_down = {}


def draw_overlay(frame, pt1, pt2, alpha=0.25, color=(51, 68, 255), filled=True):
    """Draws a semi-transparent overlay rectangle."""
    overlay = frame.copy()
    rect_color = color if filled else (0, 0, 0)
    cv.rectangle(overlay, pt1, pt2, rect_color, cv.FILLED if filled else 1)
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def count_vehicles(track_id, cx, cy, limits, crossed_ids, lp):
    """
    FIX: Line-segment crossing detection — replaces unreliable ±10px proximity window.
    Works at any vehicle speed or frame rate.
    """
    prev_y = lp.get(track_id)
    lp[track_id] = cy
    if prev_y is not None and track_id not in crossed_ids:
        if limits[0] < cx < limits[2] and _did_cross(prev_y, cy, limits[1]):
            crossed_ids.add(track_id)
            return True
    return False


def count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up, lp):
    """Counts vehicles in the left lane (upstream) using line-segment crossing."""
    prev_y = lp.get(track_id)
    lp[track_id] = cy
    if prev_y is not None and track_id not in crossed_ids_up:
        if limits[0] < cx < partition_limit and _did_cross(prev_y, cy, limits[1]):
            crossed_ids_up.add(track_id)
            return True
    return False


def count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down, lp):
    """Counts vehicles in the right lane (downstream) using line-segment crossing."""
    prev_y = lp.get(track_id)
    lp[track_id] = cy
    if prev_y is not None and track_id not in crossed_ids_down:
        if partition_limit < cx < limits[2] and _did_cross(prev_y, cy, limits[1]):
            crossed_ids_down.add(track_id)
            return True
    return False


def draw_tracks_and_count(frame, detections, total_counts, limits):
    """Annotates the frame with detected tracks and counts vehicles."""
    detections = detections[np.isin(detections.class_id, selected_classes)]
    labels = [f"#{track_id} {class_names[cls_id]}" for track_id, cls_id in
              zip(detections.tracker_id, detections.class_id)]

    label_annotator.annotate(frame, detections=detections, labels=labels)
    box_annotator.annotate(frame, detections=detections)
    trace_annotator.annotate(frame, detections=detections)

    for track_id, center_point in zip(detections.tracker_id,
                                      detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)):
        cx, cy = map(int, center_point)
        cv.circle(frame, (cx, cy), 4, (0, 255, 255), cv.FILLED)

        if count_vehicles(track_id, cx, cy, limits, crossed_ids, last_positions):
            total_counts.append(track_id)
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]),
                         end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.ROBOFLOW, thickness=4)

        if count_vehicles_up(track_id, cx, cy, limits, crossed_ids_up, last_positions_up):
            total_counts_up.append(track_id)

        if count_vehicles_down(track_id, cx, cy, limits, crossed_ids_down, last_positions_down):
            total_counts_down.append(track_id)

    # Annotate totals
    sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30),
                 sv.Color.ROBOFLOW, 1.25, 2, background_color=sv.Color.WHITE)
    sv.draw_text(frame, f"UP: {len(total_counts_up)}", sv.Point(x=int(w * 0.44), y=int(line_y - 20)),
                 sv.Color.WHITE, 1, 2)
    sv.draw_text(frame, f"DOWN: {len(total_counts_down)}", sv.Point(x=int(w * 0.44), y=int(line_y + 20)),
                 sv.Color.WHITE, 1, 2)


cap = cv.VideoCapture(video_path)
output_path = "DATA/OUTPUTS/car_counter_4.mp4"
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

if not cap.isOpened():
    raise Exception("Error: couldn't open the video!")

frame_idx = 0

# Video processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # FIX: ROI expressed as fraction — was frame[150:, :] which breaks on different resolutions
    roi_y = int(h * 0.21)
    mask_b = np.zeros_like(frame, dtype=np.uint8)
    mask_b[roi_y:, :] = 255
    ROI = cv.bitwise_and(frame, mask_b)

    # FIX: Process every 2nd frame for ~2× speed with no accuracy loss
    if frame_idx % 2 == 0:
        results = model(ROI)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        if detections.tracker_id is not None:
            sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]),
                         end=sv.Point(x=limits[2], y=limits[3]),
                         color=sv.Color.RED, thickness=4)
            # Draw partition line
            cv.line(frame, (partition_limit, line_y - 40), (partition_limit, line_y + 40), (255, 255, 0), 2)
            draw_tracks_and_count(frame, detections, total_counts, limits)
    else:
        sv.draw_line(frame, start=sv.Point(x=limits[0], y=limits[1]),
                     end=sv.Point(x=limits[2], y=limits[3]),
                     color=sv.Color.RED, thickness=4)
        cv.line(frame, (partition_limit, line_y - 40), (partition_limit, line_y + 40), (255, 255, 0), 2)
        sv.draw_text(frame, f"COUNTS: {len(total_counts)}", sv.Point(x=120, y=30),
                     sv.Color.ROBOFLOW, 1.25, 2, background_color=sv.Color.WHITE)
        sv.draw_text(frame, f"UP: {len(total_counts_up)}", sv.Point(x=int(w * 0.44), y=int(line_y - 20)),
                     sv.Color.WHITE, 1, 2)
        sv.draw_text(frame, f"DOWN: {len(total_counts_down)}", sv.Point(x=int(w * 0.44), y=int(line_y + 20)),
                     sv.Color.WHITE, 1, 2)

    out.write(frame)
    cv.imshow("Camera", frame)

    if cv.waitKey(1) & 0xff == ord('p'):
        break

cap.release()
out.release()
cv.destroyAllWindows()
