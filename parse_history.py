import os
import glob
import warnings
import pandas as pd
from betfair_data import bflw

warnings.filterwarnings("ignore", category=DeprecationWarning)

RAW_DIR = "historical_data/raw"
PROCESSED_FILE = "historical_data/processed/historical_features.csv"

def parse_historical_archives():
    paths = glob.glob(os.path.join(RAW_DIR, "*.tar")) + glob.glob(os.path.join(RAW_DIR, "*.zip"))
    print(f"Found {len(paths)} archive files to process...")

    if not paths:
        print("⚠️ No tar or zip archives found in historical_data/raw/.")
        return

    dataset_rows = []
    for file_obj in bflw.Files(paths):
        for market in file_obj:
            market_items = market if isinstance(market, list) else [market]
            for item in market_items:
                market_definition = getattr(item, "market_definition", None)
                if not market_definition or getattr(market_definition, "market_type", None) != "WIN":
                    continue
                
                market_id = getattr(item, "market_id", None)
                event_date = getattr(market_definition, "market_time", None)
                runners = getattr(market_definition, "runners", [])
                if runners:
                    for runner in runners:
                        dataset_rows.append({
                            "market_id": market_id,
                            "date": event_date,
                            "selection_id": getattr(runner, "id", None),
                            "bsp": getattr(runner, "bsp", None),
                            "won": 1 if getattr(runner, "status", None) == "WINNER" else 0
                        })

    if not dataset_rows:
        print("⚠️ Warning: Parsed 0 rows.")
        return

    df = pd.DataFrame(dataset_rows)
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df.to_csv(PROCESSED_FILE, index=False)
    print(f"✅ Successfully parsed historical data and saved {len(df)} rows to {PROCESSED_FILE}")

if __name__ == "__main__":
    parse_historical_archives()
