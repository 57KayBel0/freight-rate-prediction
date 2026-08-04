import json
import joblib
import pandas as pd

# ====================================================
# Load model
# ====================================================

model = joblib.load("models/catboost_tuned.joblib")

# ====================================================
# Load training medians
# ====================================================

with open("models/medians.json", "r") as f:
    medians = json.load(f)

weight_median = medians["weight_median"]
market_index_median = medians["market_index_median"]


with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ====================================================
# Load training data (used to reconstruct December data)
# ====================================================

train = pd.read_csv("data/train_clean.csv")

route = train[
    (train["pickup"] == "Lexington") &
    (train["delivery"] == "Fort Wayne")
]

pickup_lat = route["pickup_lat"].iloc[0]
pickup_lon = route["pickup_lon"].iloc[0]

delivery_lat = route["delivery_lat"].iloc[0]
delivery_lon = route["delivery_lon"].iloc[0]

market_index = route["market_index"].mean()
quote_signal = route["quote_signal"].mean()


# ====================================================
# Feature Engineering
# ====================================================

def engineer_features(df):

    df = df.copy()

    # Date features
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Route
    df["route"] = df["pickup"] + " -> " + df["delivery"]

    # Missing values
    df["weight"] = df["weight"].fillna(weight_median)
    df["market_index"] = df["market_index"].fillna(market_index_median)

    # Engineered features
    df["lat_diff"] = abs(df["pickup_lat"] - df["delivery_lat"])
    df["lon_diff"] = abs(df["pickup_lon"] - df["delivery_lon"])

    df["distance_weight_ratio"] = (
        df["distance"] / df["weight"]
    )

    df["distance_squared"] = (
        df["distance"] ** 2
    )

    df["market_distance"] = (
        df["distance"] * df["market_index"]
    )

    df["quote_distance"] = (
        df["distance"] * df["quote_signal"]
    )

    return df


# ====================================================
# Validation Predictions
# ====================================================

print("Predicting validation set...")

validation = pd.read_csv("data/validation.csv")

validation_features = engineer_features(validation)

validation_model = validation_features.drop(
    columns=[
        "load_id",
        "date"
    ]
)

validation_model = validation_model[feature_columns]

validation_predictions = model.predict(validation_model)

submission = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": validation_predictions
})

submission.to_csv(
    "validation_predictions.csv",
    index=False
)

print("✓ Saved validation_predictions.csv")


# ====================================================
# December Predictions
# ====================================================

print("Predicting December data...")

december = pd.read_csv("data/december-chart-inputs.csv")

# Add the missing features from historical Lexington -> Fort Wayne loads

december["pickup_lat"] = pickup_lat
december["pickup_lon"] = pickup_lon

december["delivery_lat"] = delivery_lat
december["delivery_lon"] = delivery_lon

december["market_index"] = market_index
december["quote_signal"] = quote_signal

# Feature engineering

december_features = engineer_features(december)

december_model = december_features.drop(
    columns=["date", "predicted_rate"]
)

december_model = december_model[feature_columns]

december_predictions = model.predict(december_model)

december["predicted_rate"] = december_predictions

submission_december = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
]

submission_december.to_csv(
    "december_predictions.csv",
    index=False
)

print("✓ Saved december_predictions.csv")

print("\nPrediction complete!")