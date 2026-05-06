from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.path.join(os.path.dirname(__file__), "tacloban_eta_rfr_model.pkl")
model = joblib.load(model_path)

@app.get("/")
async def root():
    return {"message": "Tacloban PUV ETA API is live and ready!"}

@app.post("/predict")
async def predict_eta(data: dict):
    try:
        feature_order = ["LTI_Mean", "Velocity_kmh", "Rush_Hour", "Weather", "Day_Num", "Time_Encoded"]
        df = pd.DataFrame([data])
        df = df[feature_order]
        prediction = model.predict(df)
        return {"eta_minutes": round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e), "status": "failed"}