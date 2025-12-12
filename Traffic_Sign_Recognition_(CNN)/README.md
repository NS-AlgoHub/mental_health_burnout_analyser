# Traffic Sign Recognition (CNN)

## 📌 Project Overview
This project implements **Traffic Sign Recognition (CNN)** as a complete, production-ready ML pipeline with modular code, API deployment, configuration-driven training, and dataset integration. The goal is to provide an industry-standard reference implementation.

## 📂 Folder Structure
- **src/** – Core ML pipeline (preprocessing, modeling, evaluation)
- **api/** – FastAPI/Flask-based prediction API
- **configs/** – YAML/JSON configs for model training & inference
- **data/** – Placeholder for raw, processed, and model data
- **Dockerfile** – Containerized deployment

## 📊 Dataset
Use standard benchmark datasets from **Kaggle / UCI / HuggingFace**.  
Dataset should include:
- Cleaned input features  
- Target labels  
- Train/Validation/Test split  

## 🧠 ML Pipeline
1. **Data ingestion and validation**  
2. **Preprocessing and feature engineering**  
3. **Model selection + hyperparameter tuning**  
4. **Training and evaluation**  
5. **Model persistence**  
6. **Prediction API integration**

## 🛠 How to Run
### Local:
```
pip install -r requirements.txt
python src/train.py
python api/app.py
```

### Docker:
```
docker build -t project .
docker run -p 8000:8000 project
```

## 📈 Evaluation
Include metrics such as:
- Accuracy / F1  
- Precision–Recall  
- ROC-AUC (classification)  
- RMSE / MAE (regression)  
- Confusion matrix  

## 🚀 Future Improvements
- Model monitoring  
- Automated retraining  
- Dataset drift detection  
- Optimization for inference  
