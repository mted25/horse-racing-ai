import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model():
    # Point directly to your processed model-ready dataset
    history_file = "historical_data/processed/model_ready_data.csv"

    # Optional debugging prints to verify paths in your terminal
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"🔍 Looking for historical data at: {os.path.abspath(history_file)}")

    if not os.path.exists(history_file):
        print(f"\n⚠️ Error: Could not find '{history_file}' in your workspace.")
        print("Please ensure your processed CSV file is saved here.")
        return

    # Load the historical dataset
    df = pd.read_csv(history_file)

    # Update features based on what is actually in your dataset
    # (Since your live fetch script pulls 'odds', we can use that)
    features = ["odds"]
    
    # Check that required columns exist
    for f in features:
        if f not in df.columns:
            print(f"⚠️ Warning: Missing expected column '{f}' in dataset.")

    print("🚀 Training Random Forest model from CSV data...")
    
    # Simple training implementation example:
    X = df[['odds']]
    # Creating a dummy target if 'won' column isn't explicitly in the CSV yet
    y = (df['odds'] < 5.0).astype(int) 
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save the model so Streamlit can find it
    import joblib
    joblib.dump(model, "horse_tipster_model.pkl")
    print("✅ Model successfully trained and saved as horse_tipster_model.pkl!")

if __name__ == "__main__":
    train_model()
