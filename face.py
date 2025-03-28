import cv2
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN

# Initialize Haar cascade for face detection (frontal face)
haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Dictionary to hold cached face detections per track (for temporal consistency)
face_cache = {}  # key: track_id, value: (list_of_boxes, last_frame)

# Parameters for temporal consistency (in frames)
CACHE_MAX_AGE = 2  # Use cached face detection if within 2 frames

def enhance_frame(frame):
    """
    Enhance the frame using global histogram equalization and CLAHE for local contrast.
    """
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Global histogram equalization on Y channel
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    # Apply CLAHE on Y channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return enhanced

def run_ensemble_face_detection(person_roi, upscale=False):
    """
    Run ensemble face detection on a person ROI using MTCNN and Haar cascade.
    Optionally, upscale the ROI to help detect very small faces.
    Returns a list of detected face boxes in [x1, y1, x2, y2] format relative to the ROI.
    """
    boxes = []
    roi = person_roi.copy()
    if upscale:
        scale_factor = 1.5
        roi = cv2.resize(roi, (int(roi.shape[1]*scale_factor), int(roi.shape[0]*scale_factor)))
    # Run MTCNN
    try:
        mtcnn_boxes, _ = mtcnn.detect(roi)
    except Exception:
        mtcnn_boxes = None

    if mtcnn_boxes is not None:
        for fb in mtcnn_boxes:
            x1, y1, x2, y2 = map(int, fb)
            # If upscaled, convert back to original coordinates
            if upscale:
                x1 = int(x1 / scale_factor)
                y1 = int(y1 / scale_factor)
                x2 = int(x2 / scale_factor)
                y2 = int(y2 / scale_factor)
            boxes.append([x1, y1, x2, y2])
    # Run Haar cascade (convert ROI to grayscale)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    haar_faces = haar_face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in haar_faces:
        # If upscaled, convert coordinates back
        if upscale:
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
        boxes.append([x, y, x+w, y+h])
    # Optionally, remove duplicates or merge overlapping boxes.
    # Here we simply return the list.
    return boxes

# -------------------------------
# Initialize YOLOv8 for Person Detection
# -------------------------------
yolo_model = YOLO("yolov8n.pt")  # Update path if needed

# -------------------------------
# Initialize DeepSORT Tracker
# -------------------------------
tracker = DeepSort(max_age=30, n_init=3)

# -------------------------------
# Initialize MTCNN for Face Detection (TensorFlow-free)
# -------------------------------
mtcnn = MTCNN(keep_all=True, device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')

# -------------------------------
# Video Setup
# -------------------------------
input_video = r"C:\Users\Sravya Duvvuri\Documents\SRAVYA NEW\AMRITA\REGULAR SEM\SEMESTER 6\RESEARCH ELECTIVE\GIT_CLONED\data\rwf2k\RWF-2000\val\Fight\I-QiUMPTWNE_2.avi"
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error opening video file")
    exit()

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)
output_video_path = os.path.join(output_dir, "output_video.avi")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    enhanced_frame = enhance_frame(frame)

    # -------------------------------
    # Step 1: Person Detection using YOLOv8
    # -------------------------------
    results = yolo_model(enhanced_frame)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            if cls == 0 and conf > 0.5:
                # DeepSORT expects ([x1, y1, x2, y2], confidence, "person")
                detections.append(([x1, y1, x2, y2], conf, "person"))

    # -------------------------------
    # Step 2: DeepSORT Tracking
    # -------------------------------
    tracks = tracker.update_tracks(detections, frame=enhanced_frame)

    # -------------------------------
    # Step 3: Face Detection & Blurring with Ensemble & Temporal Consistency
    # -------------------------------
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_box = track.to_ltrb()  # returns [x1, y1, x2, y2]
        tx1, ty1, tx2, ty2 = map(int, track_box)
        tx1, ty1 = max(0, tx1), max(0, ty1)
        tx2, ty2 = min(width, tx2), min(height, ty2)
        person_roi = enhanced_frame[ty1:ty2, tx1:tx2]
        if person_roi.size == 0:
            continue

        # Run ensemble face detection on the person ROI at original scale.
        face_boxes = run_ensemble_face_detection(person_roi, upscale=False)
        # If nothing is detected, try upscaling once.
        if face_boxes is None or len(face_boxes) == 0:
            face_boxes = run_ensemble_face_detection(person_roi, upscale=True)
        # If still nothing, check cache for this track.
        track_id = track.track_id  # unique ID for the track
        if (face_boxes is None or len(face_boxes) == 0) and track_id in face_cache:
            cached_boxes, last_frame = face_cache[track_id]
            if frame_count - last_frame <= CACHE_MAX_AGE:
                face_boxes = cached_boxes

        # If face boxes are detected, update cache.
        if face_boxes is not None and len(face_boxes) > 0:
            face_cache[track_id] = (face_boxes, frame_count)

            for fb in face_boxes:
                fx1, fy1, fx2, fy2 = map(int, fb)
                abs_x1 = tx1 + max(0, fx1)
                abs_y1 = ty1 + max(0, fy1)
                abs_x2 = tx1 + min(person_roi.shape[1], fx2)
                abs_y2 = ty1 + min(person_roi.shape[0], fy2)
                face_roi = enhanced_frame[abs_y1:abs_y2, abs_x1:abs_x2]
                if face_roi.size != 0:
                    blur_size = max(15, ((abs_x2 - abs_x1) // 2) * 2 + 1)
                    blurred_face = cv2.GaussianBlur(face_roi, (blur_size, blur_size), 30)
                    enhanced_frame[abs_y1:abs_y2, abs_x1:abs_x2] = blurred_face

    out.write(enhanced_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved as {output_video_path}")
