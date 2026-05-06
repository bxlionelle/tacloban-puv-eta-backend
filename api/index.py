from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

# Load the model
# Vercel handles the pathing; this ensures it finds the .pkl file
model_path = os.path.join(os.getcwd(), "tacloban_eta_rfr_model.pkl")
model = joblib.load(model_path)

@app.get("/")
def home():
    return {"status": "Tacloban PUV ETA API is Running"}

@app.post("/predict")
async def predict(data: dict):
    try:
        # Expected keys from Flutter: LTI_Mean, Velocity_kmh, Rush_Hour, Weather, Day_Num, Time_Encoded
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        return {"eta_minutes": round(prediction[0], 2)}
    except Exception as e:
        return {"error": str(e)}