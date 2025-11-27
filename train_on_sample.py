#!/usr/bin/env python3
"""
Preprocess sampled Airbnb CSV and train XGBoost + LightGBM safely.
Usage:
  python train_on_sample.py --input sampled_airbnb.csv --outdir airbnb_models
"""

import argparse, os, json, re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib


# -----------------------------------------
# Clean money-like / numeric strings
# -----------------------------------------
def to_numeric_strip(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = re.sub(r'[^0-9.\-]', ' ', x)
        m = re.search(r"[-+]?\d*\.?\d+", s)
        return float(m.group(0)) if m else np.nan
    try:
        return float(x)
    except:
        return np.nan


# -----------------------------------------
# PREPROCESSING FUNCTION (Fixed)
# -----------------------------------------
def preprocess(df):
    print("🔧 Starting preprocessing...")

    # 1. Create price column
    if "log_price" in df.columns:
        df["price"] = np.exp(pd.to_numeric(df["log_price"], errors="coerce"))
    elif "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        raise ValueError("❌ No price or log_price column found.")

    # 2. Select relevant columns
    useful_cols = [
        "property_type", "room_type", "accommodates", "bathrooms", 
        "bed_type", "cancellation_policy", "cleaning_fee", "extra_people",
        "number_of_reviews", "review_scores_rating", "latitude", "longitude",
        "neighbourhood", "bedrooms", "beds", "amenities", "price"
    ]
    useful_cols = [c for c in useful_cols if c in df.columns]
    df2 = df[useful_cols].copy()

    # 3. Clean numeric-like strings
    for col in ["cleaning_fee", "extra_people"]:
        if col in df2.columns:
            df2[col] = df2[col].apply(to_numeric_strip)

    # 4. Fill missing numeric values
    numeric_cols = [
        "accommodates", "bathrooms", "cleaning_fee", "extra_people",
        "number_of_reviews", "review_scores_rating", "latitude", 
        "longitude", "bedrooms", "beds"
    ]
    for c in numeric_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
            df2[c] = df2[c].fillna(df2[c].median())

    # 5. Parse amenities into binary features
    if "amenities" in df2.columns:
        df2["amenities"] = df2["amenities"].astype(str).str.lower()

        common_amenities = [
            "wifi", "internet", "kitchen", "air conditioning", "heating",
            "washer", "dryer", "free parking", "pool", "tv", "gym",
            "pets allowed"
        ]

        for amen in common_amenities:
            df2[f"amenity_{amen.replace(' ', '_')}"] = (
                df2["amenities"].str.contains(amen).astype(int)
            )

        df2.drop(columns=["amenities"], inplace=True)

    # 6. Reduce high-cardinality categorical features
    cat_cols = [
        "property_type", "room_type", "bed_type",
        "cancellation_policy", "neighbourhood"
    ]
    for c in cat_cols:
        if c in df2.columns:
            df2[c] = df2[c].astype(str).fillna("Unknown")
            top = df2[c].value_counts().nlargest(20).index
            df2[c] = df2[c].apply(lambda x: x if x in top else "Other")

    # 7. One-hot encode the reduced categoricals
    df2 = pd.get_dummies(df2, columns=[c for c in cat_cols if c in df2.columns], drop_first=False)

    # 8. Filter invalid prices
    df2 = df2[df2["price"] > 0].reset_index(drop=True)

    print(f"✅ Preprocessing complete → Final shape: {df2.shape}")
    return df2


# -----------------------------------------
# TRAIN MODELS
# -----------------------------------------
def train(df, outdir):
    print("\n🚀 Training models...")

    from xgboost import XGBRegressor

    # Try LightGBM import
    try:
        from lightgbm import LGBMRegressor
        lgb_available = True
    except ImportError:
        print("⚠ LightGBM not installed. Only XGBoost will train.")
        lgb_available = False

    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # XGBOOST MODEL
    # -----------------------------
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist"
    )
    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    # FIX: RMSE for old sklearn
    rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))
    mape_xgb = mean_absolute_percentage_error(y_test, pred_xgb)

    # -----------------------------
    # LIGHTGBM MODEL
    # -----------------------------
    if lgb_available:
        lgb = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6
        )
        lgb.fit(X_train, y_train)
        pred_lgb = lgb.predict(X_test)

        rmse_lgb = np.sqrt(mean_squared_error(y_test, pred_lgb))
        mape_lgb = mean_absolute_percentage_error(y_test, pred_lgb)
    else:
        rmse_lgb = None
        mape_lgb = None
        lgb = None

    best_model = "lightgbm" if rmse_lgb and rmse_lgb < rmse_xgb else "xgboost"

    # Save models
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(xgb, os.path.join(outdir, "xgb_model.pkl"))
    if lgb:
        joblib.dump(lgb, os.path.join(outdir, "lgb_model.pkl"))

    # Save metrics
    metrics = {
        "rmse_xgb": rmse_xgb,
        "mape_xgb": mape_xgb,
        "rmse_lgb": rmse_lgb,
        "mape_lgb": mape_lgb,
        "best_model": best_model
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n========================")
    print("🎯 TRAINING COMPLETE")
    print("========================")
    print(json.dumps(metrics, indent=2))

    return metrics


# -----------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="./airbnb_models")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df_processed = preprocess(df)

    metrics = train(df_processed, args.outdir)
