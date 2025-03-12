import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
# from keras._tf_keras.keras.models import load_model
import joblib

def process_img(img):
    img = np.array(img)
    img = cv2.resize(img, (128,128))
    img = img/255.0
    img = np.expand_dims(img, axis=-1)  
    img = np.expand_dims(img, axis=0)  
    
    return img

model_path = os.path.join(os.getcwd(), "artifacts", "ga_model.pkl")

router = APIRouter()
@router.api_route('/predict',methods=['GET','POST'])
async def predict(file: UploadFile=File(...)):
    if not file:
        return {"error": "No file received"}
    
    image = await file.read()
    img = Image.open(BytesIO(image)).convert("L")
    img = process_img(img)
    model = joblib.load(model_path)    
    prediction = model.predict(img)    
    # return prediction
    # print(prediction)
    age = int(round(prediction[1][0][0],0))
    gender_num = int(round(prediction[0][0][0],0))
    gender = 'Male'
    if (gender_num==1):
        gender = 'Female'
    print(age, gender)
    return {'Gender':gender, 'Age':age}

    