import os
import uvicorn
import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, APIRouter, Form
from PIL import Image
from io import BytesIO
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf


model_path = os.path.join(os.path.dirname(__file__), "artifacts", "ga_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")

# Load model
model = joblib.load(model_path)

# Danh sách nhãn
LABELS = [
    "chocolate_frappe", "peach_tea", "strawberry_frappe", "strawberry_tea",
    "cold_brew", "vn_bacxiu_coffee", "vn_black_coffee", "blueberry_yogurt",
    "butterflypeaflower_bubbletea", "matcha_bubbletea", "matcha_latte",
    "original_bubbletea", "salted_foam_coffee", "strawberry_yogurt"
]

def process_img(image):
    """Xử lý ảnh trước khi đưa vào model"""
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    print('it gets here')
    image_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(image_arr, axis=0)
    return img_bat

router = APIRouter()
@router.get('/')
async def hello():
    return "hello"


@router.post('/predict')
async def predict(file: UploadFile = File(...)):  # ⚠️ Dùng File(...) thay vì File(None)

    """API nhận ảnh, xử lý và dự đoán món nước"""
    if file is None:
        return {"error": "No file received - Hãy kiểm tra request!"}

    try:
        # Đọc ảnh từ file
        image = await file.read()
        img = Image.open(BytesIO(image)).convert("RGB")
        img = process_img(img)
        # Dự đoán với model
        prediction = model.predict(img)
        confidence = float(np.max(prediction)) * 100  # Độ tự tin %
        label_index = int(np.argmax(prediction))

        # Kiểm tra index hợp lệ
        label = LABELS[label_index] if label_index < len(LABELS) else "unknown"

        return {"label": label, "confidence": f"{confidence:.2f}%"}
    
    except Exception as e:
        return {"error": f"Lỗi xử lý: {str(e)}"}
