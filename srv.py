from fastapi import FastAPI
import predict
import os
from fastapi.middleware.cors import CORSMiddleware



os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Chấp nhận tất cả nguồn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/hello")
async def hello():
    return {"message": "Welcome to the prediction API"}

app.include_router(predict.router)