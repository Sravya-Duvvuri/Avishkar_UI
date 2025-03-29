import cv2
import os

# Get OpenCV's default directory for haarcascades
haar_dir = cv2.data.haarcascades
print("Haarcascade directory:", haar_dir)

# List all cascade files
if os.path.exists(haar_dir):
    files = os.listdir(haar_dir)
    haar_files = [f for f in files if f.endswith('.xml')]
    print("Available Haar cascades:", haar_files)
else:
    print("Haarcascade directory not found!")