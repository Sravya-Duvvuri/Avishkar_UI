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
from collections import deque
import torch

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

print("Starting app...")
# Directory for uploaded video files
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

import torch.nn as nn
import torch.nn.functional as F

class Enhanced3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Enhanced3DCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)  # halves dimensions

        # Block 2
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)

        # Block 3
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)

        # Block 4
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)

        # Fully Connected layers with dropout
        self.adapt_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x): 
        # x: (B, C, T, H, W)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = self.adapt_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 3D CNN Model loading using checkpoint
model = Enhanced3DCNN(num_classes=2)  # Create an instance of your architecture
checkpoint = torch.load("model_checkpoint_epoch_70.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Haar Cascade Initialization with Check

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")

haar_face_cascade = cv2.CascadeClassifier(cascade_path)
if haar_face_cascade.empty():
    raise ValueError("Failed to load Haar cascade classifier. Check the file path.")


# Global Model Initialization

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


# Helper Functions

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

def generate_stream_3dcnn(filename):
    """
    Generator that reads the uploaded video file and processes it in segments.
    For videos at least 60 seconds long, it processes 16 uniformly sampled frames from
    each one-minute segment. For shorter videos, it samples 16 equally spaced frames from
    the entire video.
    Each segment is run through the 3D CNN model and the prediction overlay is added.
    """
    video_path = os.path.join(UPLOAD_DIR, filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        print("Invalid video parameters.")
        cap.release()
        return

    video_duration = total_frames / fps  # in seconds

    segments = []
    if video_duration >= 60:
        # Divide video into one-minute segments
        num_minutes = int(video_duration // 60)
        for i in range(num_minutes):
            start_frame = int(i * 60 * fps)
            end_frame = int(min((i+1) * 60 * fps, total_frames))
            segments.append((start_frame, end_frame))
        # Optionally, add the last partial minute segment if any
        if total_frames > num_minutes * 60 * fps:
            segments.append((int(num_minutes * 60 * fps), total_frames))
    else:
        # Only one segment: the entire video
        segments.append((0, total_frames))

    for (start_frame, end_frame) in segments:
        # Sample 16 uniformly spaced frame indices from this segment
        num_segment_frames = end_frame - start_frame
        if num_segment_frames < 16:
            indices = np.linspace(start_frame, end_frame - 1, num=16, dtype=int)
        else:
            indices = np.linspace(start_frame, end_frame - 1, num=16, dtype=int)
        
        clip_frames = []
        # For each index, set the position and read the frame
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # If we fail to read, use a black frame
                frame = np.zeros((112,112,3), dtype=np.uint8)
            clip_frames.append(frame)
        
        # Preprocess frames for the 3D CNN: convert to RGB and resize to (112,112)
        processed_clip = []
        for frame in clip_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (112, 112))
            processed_clip.append(resized)
        clip = np.stack(processed_clip, axis=0)   # shape: (T, H, W, C)
        clip = clip.transpose(3, 0, 1, 2)           # shape: (C, T, H, W)
        clip = clip / 255.0                         # normalize to [0,1]
        input_tensor = torch.from_numpy(clip).float().unsqueeze(0)  # (1, C, T, H, W)

        # Run inference on the sampled clip
        with torch.no_grad():
            outputs = model(input_tensor)
            pred_prob = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(pred_prob, dim=1).item()
            confidence = pred_prob[0, pred_class].item()
            # If confidence is low (<0.5), process frames using face blurring
            if confidence < 0.6:
                # Re-run each frame through the face anonymization pipeline
                processed_frames = []
                for frame in clip_frames:
                    processed_frames.append(process_frame(frame))
                # You can also update stage_text to show that face anonymization is applied.
                stage_text = f"Low confidence ({confidence:.2f}). Applying face anonymization."
            else:
                label = "Violence Detected" if pred_class == 1 else "No Violence Detected"
                stage_text = f"Prediction: {label} (Confidence: {confidence:.2f})"
        # For visualization, overlay the prediction text on each frame of the segment.
        for frame in clip_frames:
            overlay_frame = frame.copy()
            overlay_frame = cv2.putText(overlay_frame, stage_text, (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, jpeg = cv2.imencode('.jpg', overlay_frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()


# Flask Routes (All Pages)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/face-anonymizer')
def face_anonymizer():
    return render_template('face-anonymizer.html')


# Routes for Face Anonymizer Functionality

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
            # Yield one blank frame to clear the output, then break
            blank_frame = np.zeros((480, 1280, 3), dtype=np.uint8)
            ret, jpeg = cv2.imencode('.jpg', blank_frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
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
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    cap.release()

# Endpoint to Stop Uploaded Video Processing

@app.route('/stop_processing', methods=['POST'])
def stop_processing_video():
    data = request.get_json() or {}
    # If a reset flag is provided, clear the stop event
    if data.get("reset", False):
        stop_processing_event.clear()
        return jsonify({"message": "Stop state reset"}), 200
    stop_processing_event.set()
    return jsonify({"message": "Processing stopped"}), 200


# Socket.IO for Realtime Webcam Processing
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

# Flask Routes for 3D CNN Demo
@app.route('/CNN_3D')
def CNN_3D():
    # Render a template with an upload form and detection button
    return render_template('CNN_3D.html')

@app.route('/upload_3dcnn', methods=['POST'])
def upload_video_3dcnn():
    file = request.files.get('video')
    if not file:
        return "No video uploaded", 400
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)
    return jsonify({'stream_url': url_for('process_video_stream_3dcnn', filename=filename)})

@app.route('/process_video_3dcnn/<filename>')
def process_video_stream_3dcnn(filename):
    return Response(generate_stream_3dcnn(filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True)
