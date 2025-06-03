from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import gcsfs
import os

# -------------------------------------
# 🔧 GCS CONFIG & MODEL DOWNLOAD
# -------------------------------------
BUCKET_NAME = "fashionclip-api"
MODEL_PATH = f"{BUCKET_NAME}/cv_model/yolov8n.pt"

fs = gcsfs.GCSFileSystem()

if not os.path.exists('yolov8n.pt'):
    print("🔽 Downloading YOLOv8 model from GCS...")
    with fs.open(MODEL_PATH, 'rb') as f:
        with open('yolov8n.pt', 'wb') as local_file:
            local_file.write(f.read())
    print("✅ Model downloaded and saved as yolov8n.pt")
else:
    print("✅ Model already exists locally. Skipping download.")

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# -------------------------------------
# ✨ SHARPENING UTILITIES
# -------------------------------------
def apply_sharpening(image):
    img = np.array(image)
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, sharpening_kernel)
    return Image.fromarray(sharpened)

def calculate_brightness(image):
    grayscale = np.array(image.convert("L"))
    return np.mean(grayscale)

def should_apply_sharpening(image, threshold=115):
    brightness = calculate_brightness(image)
    print(f"🌒 Brightness of crop: {brightness:.2f}")
    return brightness < threshold

# -------------------------------------
# 🧠 MAIN FUNCTION: Person Detection + Optional Sharpening
# -------------------------------------
def process_image_cv(pil_img):
    img = np.array(pil_img)

    results = model(img)

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # class 0 = 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img[y1:y2, x1:x2]
                cropped_pil = Image.fromarray(cropped_img)

                # Optional sharpening based on brightness
                if should_apply_sharpening(cropped_pil):
                    cropped_pil = apply_sharpening(cropped_pil)

                return cropped_pil

    return pil_img  # Return original if no person detected
