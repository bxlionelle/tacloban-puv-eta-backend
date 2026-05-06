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
    # Convert incoming JSON data into a Pandas DataFrame
    df = pd.DataFrame([data])
    
    # Generate prediction from the Random Forest model
    prediction = model.predict(df)
    
    # Return the result as JSON
    return {"eta_minutes": round(prediction[0], 2)}