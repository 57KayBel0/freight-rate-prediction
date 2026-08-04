import pandas as pd
import numpy as np
import joblib
import json

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

# Training medians
weight_median = train["weight"].median()
market_index_median = train["market_index"].median()

# ----------------------------
# Feature Engineering
# ----------------------------

train["lat_diff"] = abs(train["pickup_lat"] - train["delivery_lat"])
train["lon_diff"] = abs(train["pickup_lon"] - train["delivery_lon"])
train["distance_weight_ratio"] = train["distance"] / train["weight"]

train["distance_squared"] = train["distance"] ** 2
train["market_distance"] = train["distance"] * train["market_index"]
train["quote_distance"] = train["distance"] * train["quote_signal"]

train["weight"] = train["weight"].fillna(weight_median)
train["market_index"] = train["market_index"].fillna(market_index_median)

# ----------------------------
# Features
# ----------------------------

X = train.drop(columns=["posted_rate", "load_id", "date"])
y = train["posted_rate"]

# Save feature order
feature_columns = X.columns.tolist()

with open("models/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

print("Saved feature columns.")

# ----------------------------
# Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# ----------------------------
# CatBoost categorical columns
# ----------------------------

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

# ----------------------------
# Model
# ----------------------------

model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    random_seed=42,
    loss_function="RMSE",
    eval_metric="RMSE",
    early_stopping_rounds=200,
    verbose=100,
)

print("Training CatBoost...")

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True,
)

print("\nTraining Finished!")

medians = {
    "weight_median": float(weight_median),
    "market_index_median": float(market_index_median)
}

with open("models/medians.json", "w") as f:
    json.dump(medians, f)

print("Saved medians.")

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"\nMAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

joblib.dump(model, "models/catboost_tuned.joblib")