
from fastapi import FastAPI
import pickle, pandas as pd

app = FastAPI()
model = pickle.load(open("models/model.pkl","rb"))

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}
