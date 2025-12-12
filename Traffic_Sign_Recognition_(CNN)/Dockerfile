
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install scikit-learn pandas fastapi uvicorn mlflow pyyaml
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
