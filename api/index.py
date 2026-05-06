from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()

# Just load the model normally
model_path = os.path.join(os.path.dirname(__file__), "..", "tacloban_eta_rfr_model.pkl")
model = joblib.load(model_path)

@app.post("/predict")
async def predict_eta(data: dict):
    # No Sui check needed here anymore
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return {"eta_minutes": round(prediction[0], 2)}