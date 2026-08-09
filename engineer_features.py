import os
import pandas as pd
import numpy as np

INPUT_FILE = "historical_data/processed/historical_features.csv"
OUTPUT_FILE = "historical_data/processed/model_ready_data.csv"

def engineer_features():
    if not os.path.exists(INPUT_FILE):
        print(f"⚠️ Error: {INPUT_FILE} not found. Run parse_history.py first.")
        return

    print(f"Loading historical features from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    df = df.dropna(subset=["bsp"])
    df["bsp"] = pd.to_numeric(df["bsp"], errors="coerce")
    df = df.dropna(subset=["bsp"])

    df["implied_prob"] = 1.0 / df["bsp"].clip(lower=1.01)
    df["log_bsp"] = np.log(df["bsp"].clip(lower=1.01))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Engineered features successfully and saved {len(df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    engineer_features()
