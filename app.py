from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from facenet_pytorch import MTCNN
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

print("Starting app...")

# -------------------------------
# Haar Cascade Initialization with Check
# -------------------------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")

haar_face_cascade = cv2.CascadeClassifier(cascade_path)
if haar_face_cascade.empty():
    raise ValueError("Failed to load Haar cascade classifier. Check the file path.")

# -------------------------------
# Global Model Initialization
# -------------------------------
face_cache = {}  # For temporal consistency: {track_id: (boxes, last_frame)}
CACHE_MAX_AGE = 2  # frames

yolo_model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30, n_init=3)
# No GPU available so use CPU for MTCNN.
mtcnn_device = 'cpu'
mtcnn = MTCNN(keep_all=True, device=mtcnn_device)

frame_count = 0

# Directory for uploaded video files
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global event to signal stopping of processing (for uploaded video)
stop_processing_event = threading.Event()

# -------------------------------
# Helper Functions
# -------------------------------
def enhance_frame(frame):
    """Enhance the frame using global histogram equalization and CLAHE."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def run_ensemble_face_detection(person_roi, upscale=False):
    """
    Run ensemble face detection on a person ROI using MTCNN and Haar cascade.
    Optionally upscale the ROI for detecting small faces.
    Returns a list of detected face boxes [x1, y1, x2, y2] relative to the ROI.
    """
    boxes = []
    roi = person_roi.copy()
    if roi is None or roi.size == 0:
        print("Skipping empty person ROI")
        return []
    if roi.shape[-1] != 3:
        print("ROI does not have 3 channels, skipping")
        return []
    print("Processing ROI with shape:", roi.shape)
    scale_factor = 1.5 if upscale else 1.0
    if upscale:
        roi = cv2.resize(roi, (int(roi.shape[1]*scale_factor), int(roi.shape[0]*scale_factor)))
    try:
        mtcnn_boxes, _ = mtcnn.detect(roi)
        if mtcnn_boxes is None or len(mtcnn_boxes) == 0:
            print("MTCNN detected no faces.")
            mtcnn_boxes = []
    except Exception as e:
        print("MTCNN error:", e)
        mtcnn_boxes = []

    for fb in mtcnn_boxes:
        x1, y1, x2, y2 = map(int, fb)
        if upscale:
            x1, y1, x2, y2 = int(x1 / scale_factor), int(y1 / scale_factor), int(x2 / scale_factor), int(y2 / scale_factor)
        boxes.append([x1, y1, x2, y2])
    # Only run Haar cascade if ROI is reasonably large.
    if roi.shape[0] < 30 or roi.shape[1] < 30:
        return boxes
    if not boxes:
        try:
            try:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Add minSize parameter to help avoid errors.
                haar_faces = haar_face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
            except cv2.error as e:
                print("Haar cascade error:", e)
                haar_faces = []
            for (x, y, w, h) in haar_faces:
                if upscale:
                    x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)
                boxes.append([x, y, x+w, y+h])
        except Exception as e:
            print("Haar cascade error:", e)
    return boxes

def process_frame(frame):
    """
    Process a single frame: perform enhancement, detect persons and faces,
    and blur out the detected faces. Returns the processed (blurred) frame.
    """
    global frame_count, face_cache
    if frame is None or frame.size == 0:
        print("Error: Received an empty frame!")
        return frame  # Return original frame to avoid crashes.
    frame_count += 1
    processed_frame = frame.copy()
    enhanced_frame = enhance_frame(frame.copy())
    
    # Person detection using YOLOv8
    results = yolo_model(enhanced_frame)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            if cls == 0 and conf > 0.5:
                detections.append(([x1, y1, x2, y2], conf, "person"))
    
    # Tracking using DeepSORT
    tracks = tracker.update_tracks(detections, frame=enhanced_frame)
    height, width = frame.shape[:2]
    for track in tracks:
        if not track.is_confirmed():
            continue
        tx1, ty1, tx2, ty2 = map(int, track.to_ltrb())
        tx1, ty1 = max(0, tx1), max(0, ty1)
        tx2, ty2 = min(width, tx2), min(height, ty2)
        person_roi = enhanced_frame[ty1:ty2, tx1:tx2]
        if person_roi.size == 0:
            continue

        face_boxes = run_ensemble_face_detection(person_roi, upscale=False)
        if not face_boxes:
            face_boxes = run_ensemble_face_detection(person_roi, upscale=True)
        track_id = track.track_id
        if (not face_boxes or len(face_boxes) == 0) and track_id in face_cache:
            cached_boxes, last_frame = face_cache[track_id]
            if frame_count - last_frame <= CACHE_MAX_AGE:
                face_boxes = cached_boxes
        if face_boxes:
            face_cache[track_id] = (face_boxes, frame_count)
            for fb in face_boxes:
                fx1, fy1, fx2, fy2 = map(int, fb)
                abs_x1 = tx1 + max(0, fx1)
                abs_y1 = ty1 + max(0, fy1)
                abs_x2 = tx1 + min(person_roi.shape[1], fx2)
                abs_y2 = ty1 + min(person_roi.shape[0], fy2)
                face_roi = processed_frame[abs_y1:abs_y2, abs_x1:abs_x2]
                if face_roi.size != 0:
                    blur_size = max(15, ((abs_x2 - abs_x1) // 2) * 2 + 1)
                    blurred_face = cv2.GaussianBlur(face_roi, (blur_size, blur_size), 30)
                    processed_frame[abs_y1:abs_y2, abs_x1:abs_x2] = blurred_face
                else:
                    print("Warning: Face ROI is empty or invalid")
    return processed_frame

def generate_stream(filename):
    """
    Generator that reads the uploaded video file frame-by-frame,
    processes each frame, and returns an MJPEG stream.
    The output frame is a side-by-side comparison of the original (non-blurred)
    and processed (blurred) frames, with a labeled banner on top.
    Processing stops immediately if stop_processing_event is set.
    """
    # Clear any previous stop event at start
    stop_processing_event.clear()
    video_path = os.path.join(UPLOAD_DIR, filename)
    cap = cv2.VideoCapture(video_path)
    banner_height = 40  # Height of the label banner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    banner_color = (0, 0, 0)        # Black background

    while cap.isOpened():
        if stop_processing_event.is_set():
            print("Stop processing event set. Exiting video stream.")
            break
        ret, frame = cap.read()
        if not ret:
            break
        original_frame = frame.copy()
        processed_frame = process_frame(frame.copy())
        combined = np.hstack((original_frame, processed_frame))
        banner = np.full((banner_height, combined.shape[1], 3), banner_color, dtype=np.uint8)
        half_width = combined.shape[1] // 2
        (orig_text_w, orig_text_h), _ = cv2.getTextSize("Original", font, font_scale, font_thickness)
        (proc_text_w, proc_text_h), _ = cv2.getTextSize("Processed", font, font_scale, font_thickness)
        orig_x = (half_width - orig_text_w) // 2
        orig_y = (banner_height + orig_text_h) // 2 - 5
        proc_x = half_width + (half_width - proc_text_w) // 2
        proc_y = orig_y
        cv2.putText(banner, "Original", (orig_x, orig_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(banner, "Processed", (proc_x, proc_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        final_frame = np.vstack((banner, combined))
        ret, jpeg = cv2.imencode('.jpg', final_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

# -------------------------------
# Flask Routes (All Pages)
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/offloading')
def offloading():
    return render_template('offloading.html')

@app.route('/face-anonymizer')
def face_anonymizer():
    return render_template('face-anonymizer.html')

@app.route('/yolo')
def yolo_page():
    return render_template('yolo.html')

# -------------------------------
# Routes for Face Anonymizer Functionality
# -------------------------------
@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files.get('video')
    if not file:
        return "No video uploaded", 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    return redirect(url_for('process_video_stream', filename=filename))

@app.route('/process_video/<filename>')
def process_video_stream(filename):
    return Response(generate_stream(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------------------
# Endpoint to Stop Uploaded Video Processing
# -------------------------------
@app.route('/stop_processing', methods=['POST'])
def stop_processing_video():
    """Stops the uploaded video processing immediately"""
    stop_processing_event.set()
    return jsonify({"message": "Processing stopped"}), 200

# -------------------------------
# Socket.IO for Realtime Webcam Processing
# -------------------------------
@socketio.on('frame')
def handle_frame(data):
    try:
        header, encoded = data['image'].split(',', 1)
        decoded = base64.b64decode(encoded)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error decoding realtime frame:", e)
        return
    processed_frame = process_frame(frame.copy())
    ret, jpeg = cv2.imencode('.jpg', processed_frame)
    if not ret:
        print("Error encoding frame")
        return
    b64_encoded = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    data_url = 'data:image/jpeg;base64,' + b64_encoded
    emit('processed_frame', {'image': data_url})

if __name__ == '__main__':
    socketio.run(app, debug=True)
