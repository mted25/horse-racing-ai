import os
import numpy as np
import pandas as pd

def parse_form_string_advanced(form_str):
    """
    Advanced Recency-Weighted Form Parser:
    Gives exponential decay or heavier weight to the most recent performances.
    Characters are evaluated from right (most recent) to left (older).
    """
    if pd.isna(form_str) or str(form_str).strip() in ["", "0", "nan", "None"]:
        return 0.5  
    
    chars = [c for c in str(form_str).upper() if c.isalnum()][-4:]
    if not chars:
        return 0.5

    score = 0.0
    total_weight = 0.0
    
    # Recency weights: most recent character gets highest weight
    weights = [0.1, 0.2, 0.3, 0.4][-len(chars):]
    # Re-normalize weights to sum to 1.0
    weights = [w / sum(weights) for w in weights]
    
    for char, w in zip(chars, weights):
        char_val = 0.5
        if char == '1':
            char_val = 1.0
        elif char in ['2', '3']:
            char_val = 0.75
        elif char in ['4', '5', '6']:
            char_val = 0.4
        elif char in ['P', 'U', 'F', 'BD', 'RO']:
            char_val = 0.0
            
        score += char_val * w
        total_weight += w
        
    return (score / total_weight) if total_weight > 0 else 0.5

def train_and_score_models():
    input_path = "historical_data/processed/model_ready_data.csv"
    if not os.path.exists(input_path):
        input_path = "horse_data.csv"
        
    if not os.path.exists(input_path):
        print("⚠️ No data file found to train model.")
        return

    df = pd.read_csv(input_path)
    if df.empty:
        print("⚠️ Dataset is empty.")
        return

    print(f"📊 Processing {len(df)} runners with Enhanced AI Engine & Racing Post Ratings...")

    # 1. Process Market Odds & Apply Overround Simulation
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(15.0)
    df["odds"] = df["odds"].apply(lambda x: max(1.01, x))
    
    market_overround = 1.20
    raw_exchange_prob = 1.0 / df["odds"]
    df["odds"] = 1.0 / (raw_exchange_prob * market_overround)
    df["odds"] = df["odds"].apply(lambda x: max(1.01, x))
    df["market_implied_prob"] = 1.0 / df["odds"]

    # 2. Process Features & Apply Recency-Weighted Form Parser
    raw_trainer_win = pd.to_numeric(df.get("trainer_win_pct", 0), errors="coerce").fillna(0.0) / 100.0
    trainer_runs = pd.to_numeric(df.get("trainer_runs", 5), errors="coerce").fillna(5.0)
    df["trainer_win_pct"] = raw_trainer_win * (trainer_runs / (trainer_runs + 5.0))

    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(4.0)
    df["distance_f"] = pd.to_numeric(df["distance_f"], errors="coerce").fillna(8.0)
    df["field_size"] = pd.to_numeric(df["field_size"], errors="coerce").fillna(8.0)
    df["cd_winner"] = pd.to_numeric(df["cd_winner"], errors="coerce").fillna(0).apply(lambda x: 1.0 if x > 0 else 0.0)
    
    # Clean and parse Racing Post performance ratings & deltas
    for col in ["rp_rpr", "rp_or", "rp_ts", "rpr_vs_or"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Use the advanced recency parser
    df["parsed_form_score"] = df["form_string"].apply(parse_form_string_advanced)

    processed_races = []
    group_cols = [col for col in ["location", "off_time"] if col in df.columns]

    for name, group in df.groupby(group_cols if group_cols else ["location"]):
        g = group.copy()
        field_count = len(g)
        
        # Normalize Trainer Form Win % within the race field
        tr_max = g["trainer_win_pct"].max()
        tr_min = g["trainer_win_pct"].min()
        if tr_max > tr_min:
            g["norm_trainer"] = (g["trainer_win_pct"] - tr_min) / (tr_max - tr_min)
        else:
            g["norm_trainer"] = 0.5

        # Normalize RPR vs OR Delta within the race field (Value Indicator)
        delta_max = g["rpr_vs_or"].max()
        delta_min = g["rpr_vs_or"].min()
        if delta_max > delta_min:
            g["norm_rpr_delta"] = (g["rpr_vs_or"] - delta_min) / (delta_max - delta_min)
        else:
            g["norm_rpr_delta"] = 0.5

        # Normalize Absolute RPR within the race field (Class Indicator)
        rpr_max = g["rp_rpr"].max()
        rpr_min = g["rp_rpr"].min()
        if rpr_max > rpr_min:
            g["norm_rpr"] = (g["rp_rpr"] - rpr_min) / (rpr_max - rpr_min)
        else:
            g["norm_rpr"] = 0.5

        # --- INDEPENDENT AI-FIRST BLENDING ---
        ai_weight, market_weight = 0.70, 0.30

        # 1. Calculate raw market implied percentage out of 100
        g["market_pct"] = (1.0 / g["odds"]) * 100.0
        
        # 2. Calculate AI fundamental score incorporating form, trainer, C&D, and Racing Post ratings
        ai_raw_score = (
            (0.25 * g["norm_trainer"]) +
            (0.25 * g["parsed_form_score"]) +
            (0.15 * g["cd_winner"]) +
            (0.20 * g["norm_rpr_delta"]) +
            (0.15 * g["norm_rpr"])
        )
        ai_sum = ai_raw_score.sum()
        if ai_sum > 0:
            g["ai_pct"] = (ai_raw_score / ai_sum) * 100.0
        else:
            g["ai_pct"] = 100.0 / field_count

        # 3. Apply Independent AI-First Blend
        g["blended_win_pct"] = (market_weight * g["market_pct"]) + (ai_weight * g["ai_pct"])

        # 4. HARD CLAMP: If odds are >= 15.0, strictly cap win pct at max 1.5x market odds
        longshot_mask = g["odds"] >= 15.0
        max_allowed_pct = g["market_pct"] * 1.5
        g["blended_win_pct"] = np.where(longshot_mask, np.minimum(g["blended_win_pct"], max_allowed_pct), g["blended_win_pct"])

        # 5. Normalize final win probabilities so the whole race totals exactly 100%
        blended_sum = g["blended_win_pct"].sum()
        if blended_sum > 0:
            g["win_probability"] = g["blended_win_pct"] / blended_sum
        else:
            g["win_probability"] = 1.0 / field_count

        # Place Probability calculation based on win probability field distribution
        g["place_probability"] = (g["win_probability"] * 2.2 + 0.1).clip(upper=0.85)
        
        # --- AI FAIR ODDS & EXACT UK INDUSTRY PLACE TERMS ---
        g["ai_fair_odds"] = np.where(g["win_probability"] > 0, 1.0 / g["win_probability"], 99.0)

        if field_count >= 16:
            place_fraction = 0.25  # 1/4 odds for 4 places
        elif field_count >= 8:
            place_fraction = 0.20  # 1/5 odds for 3 places
        elif field_count >= 5:
            place_fraction = 0.25  # 1/4 odds for 2 places
        else:
            place_fraction = 0.33  # 1/3 odds for smaller fields

        est_place_decimal = 1.0 + ((g["odds"] - 1.0) * place_fraction)

        # Win EV & Place EV formulas
        g["win_ev"] = (g["win_probability"] * g["odds"]) - 1.0
        g["place_ev"] = (g["place_probability"] * est_place_decimal) - 1.0

        # Convert to final rounded percentages and metrics
        g["win_percentage"] = (g["win_probability"] * 100).round(1)
        g["place_percentage"] = (g["place_probability"] * 100).round(1)
        g["win_ev"] = g["win_ev"].round(2)
        g["place_ev"] = g["place_ev"].round(2)
        g["ai_fair_odds"] = g["ai_fair_odds"].round(2)

        processed_races.append(g)

    final_df = pd.concat(processed_races, ignore_index=True)

    # Save output directly in the root directory to prevent gitignore path issues
    final_output = "model_predictions.csv"
    final_df.to_csv(final_output, index=False)
    print(f"✅ Enhanced AI model training complete with Racing Post ratings & Fair Odds for {len(final_df)} runners.")

if __name__ == "__main__":
    train_and_score_models()