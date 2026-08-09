import os
import pandas as pd

RAW_FILE = "horse_data.csv"
PROCESSED_DIR = "historical_data/processed"
PROCESSED_FILE = os.path.join(PROCESSED_DIR, "model_ready_data.csv")

def process_raw_data():
    print(f"📥 Reading raw data from {RAW_FILE}...")
    
    if not os.path.exists(RAW_FILE):
        print(f"❌ Error: {RAW_FILE} not found. Run fetch_data.py first.")
        return

    df = pd.read_csv(RAW_FILE)

    # Create the processed directory path if it doesn't exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Ensure numeric types and handle form ratings properly
    if 'odds' in df.columns:
        df['odds'] = pd.to_numeric(df['odds'], errors='coerce').fillna(8.0)
        
    # Preserve pre-race form ratings accurately from available columns
    if 'form_rating' in df.columns or 'OR' in df.columns or 'official_rating' in df.columns:
        # Check and merge rating columns if alternate names exist in raw input
        if 'OR' in df.columns and ('form_rating' not in df.columns or df['form_rating'].isnull().all()):
            df['form_rating'] = df['OR']
        elif 'official_rating' in df.columns and ('form_rating' not in df.columns or df['form_rating'].isnull().all()):
            df['form_rating'] = df['official_rating']
            
        df['form_rating'] = pd.to_numeric(df['form_rating'], errors='coerce').fillna(100.0)
    else:
        df['form_rating'] = 100.0

    # Save the processed file to the target destination
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Successfully saved processed data to {PROCESSED_FILE}!")

if __name__ == "__main__":
    process_raw_data()