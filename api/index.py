from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os

app = FastAPI()

# --- 1. ENABLE CORS ---
# This allows your Flutter app to talk to this API without security blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- 2. LOAD MODEL ---
# Pointing to the .pkl file we uploaded to GitHub
model_path = os.path.join(os.path.dirname(__file__), "..", "tacloban_eta_rfr_model.pkl")
model = joblib.load(model_path)

# --- 3. ROOT ROUTE ---
# If you visit the Vercel URL in your browser, you'll see this message
@app.get("/")
async def root():
    return {"message": "Tacloban PUV ETA API is live and ready!"}

# --- 4. PREDICTION ROUTE ---
@app.post("/predict")
async def predict_eta(data: dict):
    try:
        # 1. Define the EXACT order of features used during model training
        feature_order = [
            "LTI_Mean", 
            "Velocity_kmh", 
            "Rush_Hour", 
            "Weather", 
            "Day_Num", 
            "Time_Encoded"
        ]
        
        # 2. Convert to DataFrame and force the column order
        # This prevents crashes if Flutter sends the keys in a different order
        df = pd.DataFrame([data])
        df = df[feature_order]
        
        # 3. Generate prediction
        prediction = model.predict(df)
        
        # 4. Return result
        return {"eta_minutes": round(prediction[0], 2)}
        
    except Exception as e:
        # If it still crashes, this will return the actual error message to Flutter
        # which is very helpful for debugging your thesis!
        return {"error": str(e), "status": "failed"}