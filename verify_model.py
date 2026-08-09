import pandas as pd
import os

output_file = "historical_data/processed/model_predictions.csv"

if not os.path.exists(output_file):
    print("⚠️ Error: model_predictions.csv not found. Run python3 train_model.py first!")
else:
    df = pd.read_csv(output_file)
    print(f"✅ Successfully loaded {len(df)} predictions from model training.")
    
    # Find longshots (odds >= 15.0) and check their new win percentages
    longshots = df[df['odds'] >= 15.0][['location', 'off_time', 'horse_name', 'odds', 'win_percentage', 'trainer_win_pct']]
    
    if not longshots.empty:
        print("\n🔍 Sample of Longshots (Odds >= 15.0) with New AI Win Probabilities:")
        print(longshots.head(10).to_string(index=False))
    else:
        print("No longshots found in the dataset.")
