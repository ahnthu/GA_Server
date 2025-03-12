# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# from fastapi import APIRouter, File, UploadFile
# from PIL import Image
# from io import BytesIO
# import cv2
# import numpy as np
# # from keras._tf_keras.keras.models import load_model
# import joblib

# def process_img(img):
#     img = np.array(img)
#     img = cv2.resize(img, (128,128))
#     img = img/255.0
#     img = np.expand_dims(img, axis=-1)  
#     img = np.expand_dims(img, axis=0)  
    
#     return img

# model_path = os.path.join(os.getcwd(), "artifacts", "ga_model.pkl")

# router = APIRouter()
# @router.api_route('/predict',methods=['GET','POST'])
# async def predict(file: UploadFile=File(...)):
#     if not file:
#         return {"error": "No file received"}
    
#     image = await file.read()
#     img = Image.open(BytesIO(image)).convert("L")
#     img = process_img(img)
#     model = joblib.load(model_path)    
#     prediction = model.predict(img)    
#     # return prediction
#     # print(prediction)
#     age = int(round(prediction[1][0][0],0))
#     gender_num = int(round(prediction[0][0][0],0))
#     gender = 'Male'
#     if (gender_num==1):
#         gender = 'Female'
#     print(age, gender)
#     return {'Gender':gender, 'Age':age}


import os
import uvicorn
import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, APIRouter, Form
from PIL import Image
from io import BytesIO

# Định nghĩa đường dẫn model
model_path = os.path.join(os.path.dirname(__file__), "artifacts", "ga_model.pkl")

# Kiểm tra model tồn tại không
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

def process_img(img):
    """Xử lý ảnh trước khi đưa vào model"""
    img = np.array(img)
    if len(img.shape) == 2:  # Nếu ảnh xám, chuyển thành RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Chuẩn hóa pixel
    img = np.expand_dims(img, axis=0)
    return img

router = APIRouter()

@router.post('/predict')
async def predict(file: UploadFile = File(None)):  # ⚠️ Dùng File(...) thay vì File(None)

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
