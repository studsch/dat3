import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/diamonds.csv")
CATEGORY_COLS = [
    "cut",
    "color",
    "clarity",
]
df[CATEGORY_COLS] = df[CATEGORY_COLS].apply(LabelEncoder().fit_transform)
x = df.drop(["price"], axis=1)
y = df["price"]

ros = RandomOverSampler()
q, w = ros.fit_resample(x, y)
q["price"] = w

q.to_csv("data/diamonds.csv", index=False)
