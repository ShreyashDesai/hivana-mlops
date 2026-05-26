# Anomaly Detection API

Live API: `https://Shreyash178-hivana-mlops.hf.space`

## Endpoints
- `GET /health` – Check API status
- `POST /predict` – Detect anomaly in 5 features
- `GET /metrics` – View prediction statistics

## Example
```bash
curl -X POST https://Shreyash178-hivana-mlops.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, -0.2, 1.2, -0.5]}'
Tech Stack
FastAPI

scikit-learn (Isolation Forest)

MLflow

Docker

Deployed on Hugging Face Spaces
