import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import numpy as np

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

train = pd.read_csv("data/train_clean.csv")

# Drop columns not used for training
X = train.drop(columns=["posted_rate", "load_id", "date"])

y = train["posted_rate"]

# ----------------------------------------------------
# Feature lists
# ----------------------------------------------------

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

numerical_features = [
    c for c in X.columns
    if c not in categorical_features
]

# ----------------------------------------------------
# Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------

numeric_transformer = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    [
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore"),
        )
    ]
)

preprocessor = ColumnTransformer(
    [
        (
            "num",
            numeric_transformer,
            numerical_features,
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

# ----------------------------------------------------
# Model
# ----------------------------------------------------

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Training...")

model.fit(X_train, y_train)

print("Finished!")

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

rmse = np.sqrt(mean_squared_error(y_test, pred))

r2 = r2_score(y_test, pred)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

joblib.dump(model, "models/random_forest.joblib")