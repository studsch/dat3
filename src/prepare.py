import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/diamonds.csv")
CATEGORY_COLS = [
    "cut",
    "color",
    "clarity",
]
df[CATEGORY_COLS] = df [CATEGORY_COLS].apply(LabelEncoder().fit_transform)

x = df.drop(["price"], axis=1)
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

x_train.to_csv("data/x_train.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)

x_test.to_csv("data/x_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)
