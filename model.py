import os
import mlflow
import argparse
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score

import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd

def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL , sep=";")
    except Exception as e:
        raise e    
def eval(actual , pred):
    rmse = mean_squared_error(pred , actual , squared=False)
    mae = mean_absolute_error(actual , pred)
    r2 = r2_score(actual , pred)
    
    return rmse , mae , r2

def main(alpha , l1_ratio):
    df = load_data()
    target = "quality"
    x = df.drop(columns = target)
    y = df[target]
    
    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 , random_state=42)
    
    mlflow.set_experiment("ML model-1")
    with mlflow.start_run():
        mlflow.log_param("alpha" , alpha)
        mlflow.log_param("l1_ratio" , l1_ratio)
        
        model = ElasticNet(alpha=alpha , l1_ratio=l1_ratio , random_state=6)
        model.fit(x_train , y_train)
        preds = model.predict(x_test)
        rmse , mae , r2 = eval(y_test , preds)
        
        mlflow.log_metric("rmse" , rmse)
        mlflow.log_metric("mae" , mae)
        mlflow.log_metric("r2 Score" , r2)
        mlflow.sklearn.log_model(model , "trained_model")
  

def eval(p1, p2):
    output_metric = p1**2 + p2**2
    return output_metric

def main(inp1, inp2):
    mlflow.set_experiment("Demo Experiment")
    with mlflow.start_run():
        mlflow.set_tag("version","1.0.0")
        mlflow.log_param("Param 1", inp1)
        mlflow.log_param("Param 2", inp2)
        metric = eval(p1=inp1, p2=inp2)
        mlflow.log_metric("Eval Metric", metric)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/example.txt", "wt") as f:
            f.write(f"Artifact Created {time.asctime()}")
        mlflow.log_artifact("dummy/example.txt")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha", "-a", type=float, default=0.2)
    args.add_argument("--l1_ratio", "-l1", type=float, default=0.3)
    parsed_args = args.parse_args()
    
    main(parsed_args.alpha, parsed_args.l1_ratio)