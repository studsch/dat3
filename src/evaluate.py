import json
import pandas as pd
from joblib import load
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


x_test = pd.read_csv("data/x_test.csv")
y_test = pd.read_csv("data/y_test.csv")

models = [
    "KNeighborsRegressor",
    "LinearRegression",
    "RandomForestRegressor",
]

for model_name in models:
    m = load(f"models/{model_name}.joblib")
    y_test_pred = m.predict(x_test)
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    with open(f"scores/{model_name}.json", "w") as f:
        json.dump({
            "mean_squared_error": f"{mse:.2f}",
            "mean_absolute_error": f"{mae:.2f}",
            "r2_score": f"{r2:.2f}",
        }, f)
