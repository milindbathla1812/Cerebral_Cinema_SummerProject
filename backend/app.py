from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os

from inference import predict

app = FastAPI(title="Cerebral Cinema API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Change later to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict_brain_activity(

    text: UploadFile = File(...),

    video: UploadFile = File(...),

    fmri: UploadFile = File(...)

):

    text_path = os.path.join(
        UPLOAD_DIR,
        text.filename
    )

    video_path = os.path.join(
        UPLOAD_DIR,
        video.filename
    )

    fmri_path = os.path.join(
        UPLOAD_DIR,
        fmri.filename
    )

    with open(text_path,"wb") as buffer:
        shutil.copyfileobj(
            text.file,
            buffer
        )

    with open(video_path,"wb") as buffer:
        shutil.copyfileobj(
            video.file,
            buffer
        )

    with open(fmri_path,"wb") as buffer:
        shutil.copyfileobj(
            fmri.file,
            buffer
        )

    print("Files uploaded successfully")

    print("Starting inference...")

    result = predict(
        text_path,
        video_path,
        fmri_path
    )

    print("Inference complete")

    return JSONResponse(
        {

            "pearson": result["pearson"],

            "graph": result["graph"]

        }

    )

@app.get("/")
def home():

    return {

        "message":"Cerebral Cinema API Running"

    }