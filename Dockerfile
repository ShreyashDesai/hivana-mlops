FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt scikit-learn numpy joblib

COPY serve_model_docker.py .

EXPOSE 8000

CMD ["uvicorn", "serve_model_docker:app", "--host", "0.0.0.0", "--port", "8000"]