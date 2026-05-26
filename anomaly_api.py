from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import IsolationForest
import mlflow
import mlflow.sklearn
from datetime import datetime

app = FastAPI(title="Anomaly Detection API")

# Train a simple model (for demo)
model = IsolationForest(contamination=0.05, random_state=42)
sample_data = np.random.randn(1000, 5)
model.fit(sample_data)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("anomaly_detection_api")

class SensorData(BaseModel):
    features: list  # 5 numbers

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
    if result == "anomaly" and pred < threshold:
        result = "normal"

    # Log to MLflow
    with mlflow.start_run(run_name=f"pred_{datetime.now().timestamp()}"):
        mlflow.log_param("input", str(data.features))
        mlflow.log_metric("anomaly", 1 if result == "anomaly" else 0)
        mlflow.log_text(result, "prediction.txt")

    return {"prediction": result, "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
