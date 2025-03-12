from fastapi import FastAPI
import predict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "Welcome to the prediction API"}

app.include_router(predict.router)