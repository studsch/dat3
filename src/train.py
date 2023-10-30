import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


x_train = pd.read_csv("data/x_train.csv")
y_train = pd.read_csv("data/y_train.csv")

models = [
    KNeighborsRegressor(),
    LinearRegression(),
    RandomForestRegressor(),
]

for m in models:
    dump(m.fit(x_train, y_train.values.ravel()), f"models/{type(m).__name__}.joblib")
