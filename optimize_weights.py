import pandas as pd
import numpy as np
import os

if not os.path.exists("historical_data/processed/model_predictions.csv"):
    print("⚠️ Error: model_predictions.csv not found.")
else:
    df = pd.read_csv("historical_data/processed/model_predictions.csv")
    
    # Dynamically find odds and probability column names
    odds_col = 'bsp' if 'bsp' in df.columns else ('odds' if 'odds' in df.columns else None)
    prob_col = 'ai_win_probability' if 'ai_win_probability' in df.columns else ('win_percentage' if 'win_percentage' in df.columns else None)
    
    print(f"DEBUG: Found columns -> Odds: {odds_col}, Win Prob: {prob_col}")
    print(f"Available columns: {list(df.columns)}")
    
    if not odds_col or not prob_col:
        print("⚠️ Error: Could not find required columns.")
    else:
        df["market_implied_prob"] = 1.0 / pd.to_numeric(df[odds_col], errors='coerce')

        best_weight = 0.5
        best_score = -float("inf")

        for w in np.linspace(0, 1, 11):
            df["blended_prob"] = (w * pd.to_numeric(df[prob_col], errors='coerce') / 100.0) + ((1 - w) * df["market_implied_prob"])
            score = -np.mean((df["blended_prob"] - df.get("won", 0)) ** 2)
            
            if score > best_score:
                best_score = score
                best_weight = w

        print(f"✅ Optimal AI Weight Split: {best_weight * 100:.0f}% AI Model / {(1 - best_weight) * 100:.0f}% Market Odds")
