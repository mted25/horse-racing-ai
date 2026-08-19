import os
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# Configuration details securely read from Environment Variables
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "the-racing-api1.p.rapidapi.com")
BETFAIR_USERNAME = os.environ.get("BETFAIR_USERNAME")
BETFAIR_PASSWORD = os.environ.get("BETFAIR_PASSWORD")
BETFAIR_APP_KEY = os.environ.get("BETFAIR_APP_KEY")

def fetch_betfair_odds():
    # Gracefully bypass Betfair interactive login on GitHub Actions cloud servers to avoid 403 WAF blocks
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("☁️ Running in GitHub Actions cloud environment. Skipping Betfair interactive login (restricted by Betfair WAF datacenter IP block).")
        return pd.DataFrame()

    print("📥 Connecting to Betfair API using interactive login...")
    betfair_rows = []
    
    if not BETFAIR_USERNAME or not BETFAIR_PASSWORD or not BETFAIR_APP_KEY:
        print("⚠️ Betfair credentials missing from environment variables. Skipping Betfair odds.")
        return pd.DataFrame(betfair_rows)

    try:
        import betfairlightweight
        from betfairlightweight import filters

        trading = betfairlightweight.APIClient(
            username=BETFAIR_USERNAME,
            password=BETFAIR_PASSWORD,
            app_key=BETFAIR_APP_KEY
        )
        
        trading.login_interactive()
        print("🔐 Betfair Login Status: SUCCESS")

        race_event_id = "7"
        now_utc = datetime.now(timezone.utc)
        end_time = now_utc + timedelta(hours=18)
        
        market_filter = filters.market_filter(
            event_type_ids=[race_event_id],
            market_countries=["GB", "IE"],
            market_type_codes=["WIN"],
            market_start_time={
                "from": now_utc.strftime("%Y-%m-%dT%TZ"),
                "to": end_time.strftime("%Y-%m-%dT%TZ")
            }
        )
        
        market_catalogue = trading.betting.list_market_catalogue(
            filter=market_filter, 
            market_projection=["RUNNER_DESCRIPTION", "EVENT"],
            max_results=200
        )
        
        print(f"📊 Found {len(market_catalogue)} active Betfair win markets for today.")
        
        runner_name_map = {}
        market_ids = []
        
        for market in market_catalogue:
            market_ids.append(market.market_id)
            if hasattr(market, 'runners') and market.runners:
                for runner in market.runners:
                    runner_name_map[runner.selection_id] = runner.runner_name.strip().lower()
        
        if market_ids:
            price_filter = filters.price_projection(
                price_data=filters.price_data(ex_best_offers=True)
            )
            
            for i in range(0, len(market_ids), 40):
                chunk = market_ids[i:i+40]
                market_books = trading.betting.list_market_book(
                    market_ids=chunk, 
                    price_projection=price_filter
                )
                
                for book in market_books:
                    if book.runners:
                        for runner in book.runners:
                            runner_name = runner_name_map.get(runner.selection_id)
                            best_price = None
                            
                            if runner.ex and runner.ex.available_to_back and len(runner.ex.available_to_back) > 0:
                                best_price = runner.ex.available_to_back[0].price
                            
                            if runner_name and best_price:
                                betfair_rows.append({
                                    "horse_name_lower": runner_name,
                                    "odds": float(best_price)
                                })
        
        trading.logout()
        print(f"✅ Successfully pulled live exchange odds for {len(betfair_rows)} runners from Betfair!")
    except Exception as e:
        print(f"⚠️ Betfair connection warning / WAF block caught: {e}")

    return pd.DataFrame(betfair_rows)

def parse_runners_from_object(race_obj, location, parsed_rows):
    off_time = race_obj.get("time", race_obj.get("off_time", "00:00"))
    going = race_obj.get("going", race_obj.get("track_condition", "Good"))
    distance = race_obj.get("distance", race_obj.get("race_distance", "1m"))
    
    raw_dist_f = race_obj.get("distance_f", 0.0)
    try:
        distance_f = float(raw_dist_f) if raw_dist_f is not None and str(raw_dist_f).lower() != "none" else 0.0
    except (ValueError, TypeError):
        distance_f = 0.0

    race_class = str(race_obj.get("race_class", "Class 4"))
    
    raw_field_size = race_obj.get("field_size", 8)
    try:
        field_size = int(raw_field_size) if raw_field_size is not None and str(raw_field_size).lower() != "none" else 8
    except (ValueError, TypeError):
        field_size = 8
    
    runners = []
    for key in ["runners", "horses", "selections", "competitors"]:
        if key in race_obj and isinstance(race_obj[key], list):
            runners = race_obj[key]
            break
            
    for idx, runner in enumerate(runners):
        horse_name = "Unknown"
        h_obj = runner.get("horse")
        
        if isinstance(h_obj, dict):
            horse_name = h_obj.get("horse_name", h_obj.get("name", "Unknown"))
        elif isinstance(h_obj, str):
            horse_name = h_obj
            
        if horse_name == "Unknown" or not horse_name:
            horse_name = runner.get("horse_name", runner.get("name", runner.get("runner_name", "Unknown")))

        raw_age = runner.get("age", 4)
        try:
            age = float(raw_age) if raw_age is not None and str(raw_age).lower() != "none" else 4.0
        except (ValueError, TypeError):
            age = 4.0

        trainer_stats = runner.get("trainer_14_days", {})
        if not isinstance(trainer_stats, dict):
            trainer_stats = {}
            
        raw_pct = trainer_stats.get("percent", 0.0)
        try:
            trainer_win_pct = float(raw_pct) if raw_pct is not None and str(raw_pct).lower() != "none" else 0.0
        except (ValueError, TypeError):
            trainer_win_pct = 0.0

        raw_runs = trainer_stats.get("runs", 0.0)
        try:
            trainer_runs = float(raw_runs) if raw_runs is not None and str(raw_runs).lower() != "none" else 0.0
        except (ValueError, TypeError):
            trainer_runs = 0.0

        form_fig = runner.get("form", runner.get("recent_form", "0"))
        
        raw_cd = runner.get("cd_winner", 0)
        try:
            cd_winner = int(raw_cd) if raw_cd is not None and str(raw_cd).lower() != "none" else 0
        except (ValueError, TypeError):
            cd_winner = 0

        # Safely extract ratings for model value comparison (OR vs RPR)
        raw_or = runner.get("ofr", runner.get("or", runner.get("official_rating", 0)))
        try:
            rp_or = float(raw_or) if raw_or is not None and str(raw_or).lower() != "none" else 0.0
        except (ValueError, TypeError):
            rp_or = 0.0

        raw_rpr = runner.get("rpr", 0)
        try:
            rp_rpr = float(raw_rpr) if raw_rpr is not None and str(raw_rpr).lower() != "none" else 0.0
        except (ValueError, TypeError):
            rp_rpr = 0.0

        raw_ts = runner.get("ts", 0)
        try:
            rp_ts = float(raw_ts) if raw_ts is not None and str(raw_ts).lower() != "none" else 0.0
        except (ValueError, TypeError):
            rp_ts = 0.0

        # Extract native odds/price if provided by API, otherwise use fallback index pricing
        raw_odds = runner.get("odds", runner.get("price", runner.get("decimal_price", None)))
        try:
            odds_val = float(raw_odds) if raw_odds is not None and str(raw_odds).lower() != "none" else float(5.0 + (idx * 2.5))
        except (ValueError, TypeError):
            odds_val = float(5.0 + (idx * 2.5))
        
        parsed_rows.append({
            "location": location,
            "off_time": off_time,
            "horse_name": str(horse_name).strip(),
            "horse_name_lower": str(horse_name).strip().lower(),
            "jockey": str(runner.get("jockey", "TBD")),
            "trainer": str(runner.get("trainer", "TBD")),
            "trainer_win_pct": trainer_win_pct,
            "trainer_runs": trainer_runs,
            "age": age,
            "distance_f": distance_f,
            "race_class": race_class,
            "field_size": field_size,
            "form_string": str(form_fig),
            "odds": odds_val,
            "status": str(runner.get("status", "Active")),
            "going": str(going),
            "distance": str(distance),
            "cd_winner": cd_winner,
            "rp_or": rp_or,
            "rp_rpr": rp_rpr,
            "rp_ts": rp_ts
        })

def extract_races_recursive(node, location, parsed_rows):
    if isinstance(node, dict):
        loc = node.get("course", node.get("meeting_name", node.get("track", location)))
        if any(k in node for k in ["runners", "horses", "selections", "competitors"]):
            parse_runners_from_object(node, loc, parsed_rows)
        for k, v in node.items():
            extract_races_recursive(v, loc, parsed_rows)
    elif isinstance(node, list):
        for item in node:
            extract_races_recursive(item, location, parsed_rows)

def fetch_rapidapi_racecards():
    if not RAPIDAPI_KEY:
        print("⚠️ RAPIDAPI_KEY missing from environment variables.")
        return pd.DataFrame()

    print("📥 Fetching live racecards & ratings metrics...")
    url = f"https://{RAPIDAPI_HOST}/v1/racecards/basic"
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
        "Content-Type": "application/json"
    }
    
    for day_query in ["today", "tomorrow"]:
        querystring = {"day": day_query}
        try:
            response = requests.get(url, headers=headers, params=querystring, timeout=15)
            if response.status_code == 200:
                data = response.json()
                parsed_rows = []
                extract_races_recursive(data, "Unknown", parsed_rows)
                
                if parsed_rows:
                    print(f"✅ Successfully pulled {len(parsed_rows)} runners with performance ratings ({day_query})!")
                    return pd.DataFrame(parsed_rows)
            else:
                print(f"⚠️ API returned status code {response.status_code} for {day_query}")
        except Exception as e:
            print(f"⚠️ API connection warning for {day_query}: {e}")

    return pd.DataFrame()

def main():
    df_rapid = fetch_rapidapi_racecards()
    df_betfair = fetch_betfair_odds()
    
    if not df_rapid.empty:
        df_combined = df_rapid
        
        # Merge Betfair odds seamlessly if available
        if not df_betfair.empty and "horse_name_lower" in df_betfair.columns:
            df_combined["horse_name_lower"] = df_combined["horse_name_lower"].str.strip().str.lower()
            df_betfair["horse_name_lower"] = df_betfair["horse_name_lower"].str.strip().str.lower()
            
            df_combined = pd.merge(df_combined, df_betfair, on="horse_name_lower", how="left", suffixes=('', '_bf'))
            if "odds_bf" in df_combined.columns:
                df_combined["odds"] = df_combined["odds_bf"].fillna(df_combined["odds"])
                df_combined.drop(columns=["odds_bf"], inplace=True, errors="ignore")
    else:
        df_combined = pd.DataFrame()

    if "horse_name_lower" in df_combined.columns:
        df_combined.drop(columns=["horse_name_lower"], inplace=True, errors="ignore")

    output_path_raw = "horse_data.csv"
    output_path_processed = "historical_data/processed/model_ready_data.csv"
    os.makedirs(os.path.dirname(output_path_processed), exist_ok=True)
    
    df_combined.to_csv(output_path_raw, index=False)
    df_combined.to_csv(output_path_processed, index=False)
    print(f"✅ Saved final dataset containing {len(df_combined)} rows with fully automated rating features ready for model training.")

if __name__ == "__main__":
    main()