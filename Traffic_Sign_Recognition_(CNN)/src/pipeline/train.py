
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle, json, mlflow
import yaml, os

def load_config():
    return yaml.safe_load(open("configs/train.yaml"))

def main():
    cfg = load_config()
    df = pd.read_csv(cfg["data"]["path"])
    X = df.drop("label",axis=1)
    y = df["label"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=cfg["data"]["test_size"])

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])

    with mlflow.start_run():
        model = LogisticRegression(max_iter=cfg["model"]["max_iter"])
        model.fit(X_train,y_train)
        acc = model.score(X_test,y_test)
        mlflow.log_metric("accuracy", acc)

        os.makedirs("models", exist_ok=True)
        pickle.dump(model, open("models/model.pkl","wb"))
        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
