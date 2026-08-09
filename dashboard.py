import os
import pandas as pd
import numpy as np
import streamlit as st

DATA_FILE = "historical_data/processed/model_predictions.csv"
FALLBACK_DATA_FILE = "historical_data/processed/model_ready_data.csv"

@st.cache_data(ttl=0)
def load_data(filepath):
    if not os.path.exists(filepath):
        if os.path.exists(FALLBACK_DATA_FILE):
            return pd.read_csv(FALLBACK_DATA_FILE)
        return pd.DataFrame()
    return pd.read_csv(filepath)

def style_fair_odds(df_view):
    """
    Applies conditional formatting to color 'ai_fair_odds':
    - Green if AI Fair Odds < BSP (Good value / Overpriced by bookies)
    - Red if AI Fair Odds > BSP (Bad value / Underpriced)
    """
    def color_logic(row):
        styles = [''] * len(row)
        try:
            fair_idx = row.index.get_loc("ai_fair_odds")
            bsp_idx = row.index.get_loc("bsp")
            fair_val = float(row.iloc[fair_idx])
            bsp_val = float(row.iloc[bsp_idx])
            
            if fair_val < bsp_val:
                styles[fair_idx] = 'color: #09ab3b; font-weight: bold;'  # Green
            elif fair_val > bsp_val:
                styles[fair_idx] = 'color: #ff4b4b; font-weight: bold;'  # Red
        except Exception:
            pass
        return styles

    return df_view.style.apply(color_logic, axis=1)

def run_dashboard():
    st.set_page_config(page_title="Horse Racing AI Tipster & Probability Finder", layout="wide")
    st.title("🐎 Horse Racing AI Tipster & Top Contenders Dashboard")
    
    df = load_data(DATA_FILE)
    if df.empty:
        st.error(f"Error: Model predictions or model-ready data not found. Please run fetch_data.py and train_model.py first.")
        return

    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    # 1. Handle Status and Odds safely
    if "status" not in df.columns:
        df["status"] = "Active"

    if "odds" in df.columns:
        df["bsp"] = pd.to_numeric(df["odds"], errors='coerce').fillna(15.0)
    elif "bsp" not in df.columns:
        df["bsp"] = 15.0

    # Ensure AI Fair Odds column exists
    if "ai_fair_odds" not in df.columns:
        df["ai_fair_odds"] = 10.0
    else:
        df["ai_fair_odds"] = pd.to_numeric(df["ai_fair_odds"], errors="coerce").fillna(10.0)

    # Ensure probabilistic metrics and fields are properly mapped from model predictions
    if "win_percentage" in df.columns:
        df["ai_win_probability"] = df["win_percentage"]
    elif "ai_win_probability" not in df.columns:
        df["ai_win_probability"] = 15.0

    if "place_percentage" in df.columns:
        df["ai_place_percentage"] = df["place_percentage"]
    elif "ai_place_percentage" not in df.columns:
        df["ai_place_percentage"] = 45.0

    # Handle EV columns cleanly
    for ev_col in ["win_ev", "place_ev"]:
        if ev_col not in df.columns:
            df[ev_col] = 0.0
        else:
            df[ev_col] = pd.to_numeric(df[ev_col], errors="coerce").fillna(0.0)

    # Ensure trainer win percentage format
    if "trainer_win_pct" in df.columns:
        df["trainer_form_%"] = (pd.to_numeric(df["trainer_win_pct"], errors="coerce").fillna(0.0) * 100).round(1)
    else:
        df["trainer_form_%"] = 0.0

    # Format C&D badge indicator for display if present
    if "cd_winner" in df.columns:
        df["horse_name_display"] = df.apply(
            lambda row: f"🏆 {row['horse_name']}" if str(row.get("cd_winner", 0)).lower() in ["true", "1", "yes", "1.0", "1.15"] else row["horse_name"],
            axis=1
        )
    else:
        df["horse_name_display"] = df["horse_name"]

    # Filter active runners for tips and strategy sections (no sidebar restrictions)
    active_df = df[df["status"].str.lower() != "non-runner"]

    # 2. Top Tips for the Day Section Focused on High Win/Place Probability & EV
    st.header("🔥 Top High-Probability Tips of the Day")
    st.write("Strongest contenders across all active meetings based on AI win probabilities, trainer form trends, market support, and expected value (EV).")
    
    top_10 = active_df.sort_values(by="ai_win_probability", ascending=False).head(10)
    
    tip_display_cols = [col for col in ["off_time", "location", "horse_name_display", "jockey", "trainer", "trainer_form_%", "bsp", "ai_fair_odds", "ai_win_probability", "ai_place_percentage", "win_ev", "place_ev"] if col in top_10.columns]
    
    top_10_view = top_10[tip_display_cols].copy()
    if "horse_name_display" in top_10_view.columns:
        top_10_view.rename(columns={"horse_name_display": "horse_name"}, inplace=True)

    formatted_top_10 = style_fair_odds(top_10_view).format({
        "trainer_form_%": "{:.1f}%",
        "ai_win_probability": "{:.1f}%",
        "ai_place_percentage": "{:.1f}%",
        "bsp": "{:.2f}",
        "ai_fair_odds": "{:.2f}",
        "win_ev": "{:+.2f}",
        "place_ev": "{:+.2f}"
    })

    st.dataframe(
        formatted_top_10,
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")

    # 3. DAILY AUTOMATED STRATEGY BETS SECTION
    st.header("🎯 Daily Automated Strategy Bets")

    # Lucky 15 Builder
    with st.expander("🍀 Recommended 4-Horse Each-Way Lucky 15 (Win-Focused)", expanded=True):
        st.markdown("*Selection logic: Positive Win EV, Fair Odds < BSP, Max Fair Odds $\\le 8.0$, and lowest fair odds in race or within 1.5 of the race minimum.*")
        
        # Filter initial base requirements for Lucky 15
        l15_base = active_df[
            (active_df["win_ev"] > 0.0) & 
            (active_df["ai_fair_odds"] < active_df["bsp"]) & 
            (active_df["ai_fair_odds"] <= 8.0)
        ].copy()
        
        # Check race-specific lowest fair odds constraints
        valid_l15_rows = []
        if not l15_base.empty and "off_time" in l15_base.columns and "location" in l15_base.columns:
            for (off_time, location), group in l15_base.groupby(["off_time", "location"]):
                min_fair_in_race = group["ai_fair_odds"].min()
                filtered_group = group[group["ai_fair_odds"] <= (min_fair_in_race + 1.5)]
                valid_l15_rows.append(filtered_group)
                
            if valid_l15_rows:
                l15_filtered = pd.concat(valid_l15_rows).sort_values(by="ai_win_probability", ascending=False)
                lucky15_candidates = l15_filtered.drop_duplicates(subset=["off_time", "location"])
                lucky15_picks = lucky15_candidates.head(4)
            else:
                lucky15_picks = pd.DataFrame()
        else:
            lucky15_picks = pd.DataFrame()
        
        if not lucky15_picks.empty and len(lucky15_picks) >= 1:
            cols = st.columns(min(len(lucky15_picks), 4))
            for idx, (_, row) in enumerate(lucky15_picks.iterrows()):
                with cols[idx]:
                    st.markdown(f"**{row['off_time']} - {row['location']}**")
                    clean_name = str(row['horse_name_display']).replace("🏆 ", "")
                    st.markdown(f"🐎 **{clean_name}**")
                    st.markdown(f"📈 Win Prob: `{row['ai_win_probability']:.1f}%`")
                    st.markdown(f"💰 Odds: `{row['bsp']:.2f}` (Fair: `{row['ai_fair_odds']:.2f}`)")
                    st.markdown(f"📊 Win EV: `{row['win_ev']:+.2f}`")
        else:
            st.info("No runners currently meet all strict Lucky 15 criteria today.")

    # Place Accumulator Builder
    with st.expander("📍 Place Accumulator (BSP 8.0 to 50.0, Place EV > 0 & Place % $\\ge 30\%$)", expanded=False):
        st.markdown("*Selection logic: BSP between 8.0 and 50.0, Place EV > 0, Place % $\\ge 30\%$, AI Fair Odds < BSP, max 1 horse per individual race.*")
        
        place_filtered = active_df[
            (active_df['bsp'] >= 8.0) & 
            (active_df['bsp'] <= 50.0) & 
            (active_df['place_ev'] > 0.0) & 
            (active_df['ai_place_percentage'] >= 30.0) & 
            (active_df['ai_fair_odds'] < active_df['bsp'])
        ].sort_values(by="ai_place_percentage", ascending=False)
        
        unique_race_places = place_filtered.groupby(['off_time', 'location']).head(1)
        place_picks = unique_race_places.head(7)
        
        if not place_picks.empty:
            display_place_df = place_picks[[
                "off_time", "location", "horse_name_display", "jockey", "trainer", "bsp", "ai_fair_odds", "ai_place_percentage", "place_ev"
            ]].copy()
            display_place_df.rename(columns={"horse_name_display": "horse_name"}, inplace=True)
            
            formatted_place = style_fair_odds(display_place_df).format({
                "bsp": "{:.2f}",
                "ai_fair_odds": "{:.2f}",
                "ai_place_percentage": "{:.1f}%",
                "place_ev": "{:+.2f}"
            })
            
            st.dataframe(
                formatted_place,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No runners match the strict Place Accumulator criteria (BSP 8-50, Place EV > 0, Place % >= 30%, Fair < BSP) today.")

    st.markdown("---")

    # 4. Meeting & Race Card Explorer with "All" Options Added
    st.header("📋 Meeting & Race Card Explorer")
    
    loc_col = "location" if "location" in df.columns else ("course" if "course" in df.columns else None)
    time_col = "off_time" if "off_time" in df.columns else None

    if loc_col and time_col:
        col1, col2 = st.columns(2)
        
        with col1:
            locations = sorted(df[loc_col].dropna().unique().tolist())
            location_options = ["All Locations"] + locations
            selected_location = st.selectbox("Select Race Course / Location:", options=location_options)
            
        if selected_location == "All Locations":
            loc_filtered_df = df
        else:
            loc_filtered_df = df[df[loc_col] == selected_location]
        
        with col2:
            race_times = sorted(loc_filtered_df[time_col].dropna().unique().tolist())
            time_options = ["All Times"] + race_times
            selected_time = st.selectbox("Select Race Time:", options=time_options)
            
        if selected_time == "All Times":
            final_race_card = loc_filtered_df
            st.subheader(f"Master Card: {selected_location}")
        else:
            final_race_card = loc_filtered_df[loc_filtered_df[time_col] == selected_time]
            st.subheader(f"Race Card: {selected_location} at {selected_time}")
            
        st.caption("Note: 🏆 indicates a confirmed Course & Distance (C&D) specialist winner.")
        
        card_display_cols = [col for col in ["off_time", "location", "horse_name_display", "jockey", "trainer", "trainer_form_%", "bsp", "ai_fair_odds", "ai_win_probability", "ai_place_percentage", "win_ev", "place_ev", "status"] if col in final_race_card.columns]
        
        card_view = final_race_card[card_display_cols].copy()
        if "horse_name_display" in card_view.columns:
            card_view.rename(columns={"horse_name_display": "horse_name"}, inplace=True)

        formatted_card = style_fair_odds(card_view).format({
            "trainer_form_%": "{:.1f}%",
            "ai_win_probability": "{:.1f}%",
            "ai_place_percentage": "{:.1f}%",
            "bsp": "{:.2f}",
            "ai_fair_odds": "{:.2f}",
            "win_ev": "{:+.2f}",
            "place_ev": "{:+.2f}"
        })

        st.dataframe(
            formatted_card,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    run_dashboard()