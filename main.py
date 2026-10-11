from pathlib import Path
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "nexdrive_mode_models.joblib"
FEATURES_PATH = MODEL_DIR / "nexdrive_sensor_columns.joblib"


# --------------------------------------------------
# 2. Load saved machine learning artifacts
# --------------------------------------------------

mode_models = joblib.load(MODEL_PATH)
sensor_columns = joblib.load(FEATURES_PATH)


# --------------------------------------------------
# 3. Create the FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="NEXDRIVE",
    description=(
        "Intelligent Vehicle Diagnostics & "
        "Sensor Anomaly Detection API"
    ),
    version="1.0.0"
)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------------------------------------------------
# 4. Define expected sensor input
# --------------------------------------------------

class VehicleData(BaseModel):
    mode: int = Field(ge=0, le=2)

    LOAD_PCT: float
    ECT: float
    MAP: float
    RPM: float = Field(ge=0)
    VSS: float = Field(ge=0)
    IAT: float
    MAF: float
    FRP: float
    BARO: float
    VPWR: float
    AAT: float


# --------------------------------------------------
# 5. Home endpoint
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# --------------------------------------------------
# 6. Vehicle anomaly prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_vehicle(data: VehicleData):

    mode = data.mode

    if mode not in mode_models:
        raise HTTPException(
            status_code=400,
            detail=f"No trained model available for Mode {mode}"
        )

    # Arrange sensor values in the exact training order
    input_data = data.model_dump(exclude={"mode"})

    input_df = pd.DataFrame(
        [[input_data[column] for column in sensor_columns]],
        columns=sensor_columns
    )

    # Select the model trained for this operating mode
    model = mode_models[mode]

    # Predict whether the sensor pattern is unusual
    prediction = model.predict(input_df)[0]

    # Higher score means a more unusual pattern
    anomaly_score = float(
        -model.score_samples(input_df)[0]
    )

    is_anomaly = prediction == -1

    return {
        "application": "NEXDRIVE",
        "mode": mode,
        "result": (
            "Anomaly Detected"
            if is_anomaly
            else "Normal Pattern"
        ),
        "anomaly_detected": bool(is_anomaly),
        "anomaly_score": round(anomaly_score, 6),
        "interpretation": (
            "This sensor pattern differs from patterns "
            "learned for this operating mode. It is not "
            "a confirmed mechanical fault."
            if is_anomaly
            else
            "This sensor pattern was not flagged as unusual. "
            "This does not guarantee that the vehicle is fault-free."
        )
    }