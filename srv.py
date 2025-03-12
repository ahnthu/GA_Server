from fastapi import FastAPI
import predict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
app = FastAPI()
@app.get("/hello")

async def hello():
    return "welcome"
app.include_router(predict.router)