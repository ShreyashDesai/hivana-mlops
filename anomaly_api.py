from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
from datetime import datetime
import csv
import os

app = FastAPI(title="Anomaly Detection API")

# Train a simple model
model = IsolationForest(contamination=0.05, random_state=42)
sample_data = np.random.randn(1000, 5)
model.fit(sample_data)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("anomaly_detection_api")

# CSV file for predictions
CSV_FILE = "predictions.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "features", "prediction"])

class SensorData(BaseModel):
    features: list

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(data: SensorData, threshold: float = 0.5):
    if len(data.features) != 5:
        raise HTTPException(400, "Need exactly 5 features")
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr)[0]
    result = "anomaly" if pred == -1 else "normal"

    # Apply threshold (example logic – you can replace with real retraining)
    if result == "anomaly" and pred < threshold:
        result = "normal"

    # Log to MLflow
    with mlflow.start_run(run_name=f"pred_{datetime.now().timestamp()}"):
        mlflow.log_param("input", str(data.features))
        mlflow.log_metric("anomaly", 1 if result == "anomaly" else 0)
        mlflow.log_text(result, "prediction.txt")

    # Save to CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), str(data.features), result])

    return {"prediction": result, "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
def metrics():
    total = 0
    anomalies = 0
    try:
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                total += 1
                if row[2] == "anomaly":
                    anomalies += 1
    except FileNotFoundError:
        pass
    return {"total_predictions": total, "anomalies": anomalies, "normals": total - anomalies}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
