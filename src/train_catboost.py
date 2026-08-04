import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from catboost import CatBoostRegressor

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

train = pd.read_csv("data/train_clean.csv")

# Feature engineering
train["lat_diff"] = abs(train["pickup_lat"] - train["delivery_lat"])
train["lon_diff"] = abs(train["pickup_lon"] - train["delivery_lon"])
train["distance_weight_ratio"] = train["distance"] / train["weight"]

# Features
X = train.drop(columns=["posted_rate", "load_id", "date"])

y = train["posted_rate"]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# CatBoost categorical columns
categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route",
]

cat_features = [
    X.columns.get_loc(col)
    for col in categorical_features
]

# ----------------------------------------------------
# Model
# ----------------------------------------------------

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=5,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
)

print("Training CatBoost...")

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True,
    early_stopping_rounds=100,
)

print("Finished!")

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"\nMAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2  : {r2:.4f}")

joblib.dump(model, "models/catboost.joblib")