import os
import pandas as pd
import numpy as np

# Automatically choose today's live data if available, otherwise look for historical features
if os.path.exists("horse_data.csv"):
    INPUT_FILE = "horse_data.csv"
elif os.path.exists("historical_data/processed/historical_features.csv"):
    INPUT_FILE = "historical_data/processed/historical_features.csv"
else:
    INPUT_FILE = None

OUTPUT_FILE = "historical_data/processed/model_ready_data.csv"

def engineer_features():
    if not INPUT_FILE:
        print("⚠️ Error: No input data file found (checked horse_data.csv and historical_features.csv).")
        return

    print(f"Loading data from {INPUT_FILE} for feature engineering...")
    df = pd.read_csv(INPUT_FILE)

    # Handle odds or bsp gracefully depending on the dataset structure
    price_col = "bsp" if "bsp" in df.columns else ("odds" if "odds" in df.columns else None)
    if price_col:
        df = df.dropna(subset=[price_col])
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.dropna(subset=[price_col])
        df["implied_prob"] = 1.0 / df[price_col].clip(lower=1.01)
        df["log_bsp"] = np.log(df[price_col].clip(lower=1.01))

    # Clean and parse Racing Post performance ratings
    for col in ["rp_rpr", "rp_or", "rp_ts"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 🚀 The Punter's Holy Grail Metric: RPR vs Official Rating Delta
    # A positive number means the horse's Racing Post ability rating is higher than its official weight rating (well-handicapped value)
    df["rpr_vs_or"] = df["rp_rpr"] - df["rp_or"]

    # Auxiliary tracking features for the model
    df["has_rpr"] = (df["rp_rpr"] > 0).astype(int)
    df["has_or"] = (df["rp_or"] > 0).astype(int)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Engineered features successfully and saved {len(df)} rows with rating deltas to {OUTPUT_FILE}")

if __name__ == "__main__":
    engineer_features()