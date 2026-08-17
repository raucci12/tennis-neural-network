"""
clean_data.py

Rebuilds "Full Dataset - Cleaned & Feature Engineering.csv" from the raw
merged WTA Grand Slam + Bet365 odds data (Full_Dataset.xlsx), following the
exact schema documented in the project's data dictionary.

Usage:
    python clean_data.py --input Full_Dataset.xlsx --output data/"Full Dataset - Cleaned & Feature Engineering.csv"

If --input/--output are omitted, sensible defaults relative to this script
are used (see DEFAULT_INPUT / DEFAULT_OUTPUT below).
"""

import argparse
import os
import re
import unicodedata
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "Full_Dataset.xlsx")
DEFAULT_OUTPUT = os.path.join(
    SCRIPT_DIR, "data", "Full Dataset - Cleaned & Feature Engineering.csv"
)
# A CSV of currently WTA-ranked players with columns: name, current_rank,
# current_pts (e.g. copy-pasted from a live rankings page). When present,
# the player dropdown in tennis_app.py is filtered to only these names AND
# enriched with each player's current rank/points for auto-fill.
# Retired/inactive players stay in the training data but are hidden from
# the dropdown. If this file doesn't exist, no filtering/enrichment happens.
# A plain .txt (one name per line, no rank/points) is also still supported
# for filtering-only, for backwards compatibility.
DEFAULT_ACTIVE_PLAYERS = os.path.join(SCRIPT_DIR, "active_players.csv")

# Known spelling variants between the historical dataset and current WTA
# rankings sources, mapped as {dataset_spelling: rankings_spelling}.
NAME_ALIASES = {
    "liudmila samsonova": "ludmilla samsonova",
}

# Country of the four WTA Grand Slam host nations, keyed by Tournament name.
TOURNAMENT_COUNTRY = {
    "Australian Open": "AUS",
    "French Open": "FRA",
    "Wimbledon": "GBR",
    "US Open": "USA",
}

# Encoding used throughout the original project (per data dictionary).
COUNTRY_CODE = {"AUS": 1, "FRA": 2, "GBR": 3, "USA": 4}
SURFACE_CODE = {"Hard": 1, "Clay": 2, "Grass": 3}
HAND_CODE = {"R": 1, "L": 2}

# Reference dates for the 2025 Grand Slams, used to project each player's
# age forward (or back) from their most recent recorded match. These are
# approximate tournament start dates -- precise enough for an age estimate.
GRAND_SLAM_2025_DATES = {
    1: pd.Timestamp("2025-01-13"),  # Australian Open
    2: pd.Timestamp("2025-05-25"),  # French Open
    3: pd.Timestamp("2025-06-30"),  # Wimbledon
    4: pd.Timestamp("2025-08-25"),  # US Open
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean and feature-engineer the raw WTA Grand Slam dataset."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                         help="Path to the raw Full_Dataset.xlsx file.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                         help="Where to write the cleaned CSV.")
    parser.add_argument("--active-players", type=str, default=DEFAULT_ACTIVE_PLAYERS,
                         help=(
                             "Optional path to a text file of currently active player "
                             "names (one per line). If present, filters the player "
                             "dropdown to only these names. Training data is unaffected."
                         ))
    return parser.parse_args()


def normalize_name(name):
    """Lowercase, strip accents/punctuation, and collapse whitespace so
    names can be matched across sources that format them differently
    (e.g. 'En-shuo Liang' vs 'En Shuo Liang')."""
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.lower()
    ascii_name = re.sub(r"[-'.]", " ", ascii_name)
    ascii_name = re.sub(r"\s+", " ", ascii_name).strip()
    return ascii_name


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Could not find raw dataset at '{args.input}'.\n"
            "Place Full_Dataset.xlsx there, or point to it explicitly with:\n"
            '  python clean_data.py --input "/path/to/Full_Dataset.xlsx"'
        )

    raw = pd.read_excel(args.input)

    # Build an ID -> name/height/age lookup for every player who appears
    # anywhere in the raw data (as either a winner or loser), before any
    # row filtering. This powers the player dropdowns in tennis_app.py,
    # including auto-filled height and age.
    player_matches = pd.concat([
        raw[["winner_id", "winner_name", "winner_age", "winner_ht", "winner_hand", "tourney_date"]]
            .rename(columns={"winner_id": "player_id", "winner_name": "name",
                              "winner_age": "age", "winner_ht": "height", "winner_hand": "hand"}),
        raw[["loser_id", "loser_name", "loser_age", "loser_ht", "loser_hand", "tourney_date"]]
            .rename(columns={"loser_id": "player_id", "loser_name": "name",
                              "loser_age": "age", "loser_ht": "height", "loser_hand": "hand"}),
    ]).dropna(subset=["player_id", "name"])

    # Name lookup: one row per player.
    players = player_matches.drop_duplicates(subset="player_id").sort_values("name")
    players = players[["player_id", "name"]].reset_index(drop=True)

    # Height: median of all recorded heights for that player (most players
    # only ever have one distinct value; median is a safe tie-breaker).
    height_by_player = (
        player_matches.dropna(subset=["height"])
        .groupby("player_id")["height"].median()
    )

    # Hand: most frequently recorded hand for that player (a handful of rows
    # have data-entry inconsistencies; the mode is a safe choice since a
    # player's dominant hand doesn't change match to match).
    valid_hand = player_matches[player_matches["hand"].isin(HAND_CODE.keys())].copy()
    valid_hand["hand_code"] = valid_hand["hand"].map(HAND_CODE)
    hand_by_player = (
        valid_hand.groupby("player_id")["hand_code"]
        .agg(lambda s: s.mode().iloc[0])
    )

    # Age: take each player's age at their most recent recorded match, then
    # project it forward/back to each 2025 Grand Slam's date.
    age_rows = player_matches.dropna(subset=["age", "tourney_date"]).copy()
    age_rows["tourney_date"] = pd.to_datetime(age_rows["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    last_known = (
        age_rows.sort_values("tourney_date")
        .groupby("player_id")
        .tail(1)
        .set_index("player_id")[["age", "tourney_date"]]
    )

    players = players.merge(height_by_player.rename("height"), on="player_id", how="left")
    players = players.merge(hand_by_player.rename("hand"), on="player_id", how="left")
    players = players.merge(last_known, on="player_id", how="left")

    # Some players never had a height recorded in any of their matches.
    # Rather than leaving those blank (forcing manual entry in the app),
    # fall back to the median height across all players who DO have one.
    median_height = height_by_player.median()
    missing_height_count = players["height"].isna().sum()
    if missing_height_count:
        print(
            f"{missing_height_count} of {len(players)} players had no recorded height "
            f"-- filled with the dataset median ({median_height:.1f} cm)."
        )
    players["height"] = players["height"].fillna(median_height)

    for code, target_date in GRAND_SLAM_2025_DATES.items():
        years_elapsed = (target_date - players["tourney_date"]).dt.days / 365.25
        players[f"age_country_{code}"] = (players["age"] + years_elapsed).round(1)

    players = players.drop(columns=["age", "tourney_date"])

    # Keep only completed matches with a known Grand Slam tournament.
    raw = raw[raw["Comment"] == "Completed"].copy()
    raw = raw[raw["Tournament"].isin(TOURNAMENT_COUNTRY.keys())].copy()

    # Drop rows with no usable date/rank/points/odds data before converting types.
    raw = raw.dropna(subset=["tourney_date", "winner_rank", "loser_rank", "WPts", "LPts", "B365W", "B365L"])

    out = pd.DataFrame()

    # --- Match Details ---
    out["tourney_date"] = pd.to_datetime(raw["tourney_date"], format="%Y%m%d").dt.strftime("%Y%m%d").astype(int)
    out["match_num"] = raw["match_num"]
    out["Surface"] = raw["Surface"].map(SURFACE_CODE)
    out["Round"] = raw["Round"]
    out["minutes"] = raw["minutes"]

    # --- Player Information (P1 = winner, P2 = loser) ---
    out["P1_id"] = raw["winner_id"]
    out["P2_id"] = raw["loser_id"]
    out["P1_hand"] = raw["winner_hand"].map(HAND_CODE)
    out["P2_hand"] = raw["loser_hand"].map(HAND_CODE)
    out["P1_ht"] = raw["winner_ht"]
    out["P2_ht"] = raw["loser_ht"]
    out["P1_age"] = raw["winner_age"]
    out["P2_age"] = raw["loser_age"]
    out["P1_Pts"] = raw["WPts"]
    out["P2_Pts"] = raw["LPts"]
    out["P1_Rank"] = raw["winner_rank"]
    out["P2_Rank"] = raw["loser_rank"]
    # tennis_cv.py's feature_engineering() reads these without the underscore
    out["P1Pts"] = out["P1_Pts"]
    out["P2Pts"] = out["P2_Pts"]
    out["P1Rank"] = out["P1_Rank"]
    out["P2Rank"] = out["P2_Rank"]

    # --- Match Statistics ---
    for col in ["W1", "L1", "W2", "L2", "W3", "L3", "Wsets", "Lsets"]:
        out[col] = raw[col]
    stat_pairs = [
        ("w_ace", "l_ace"), ("w_df", "l_df"), ("w_svpt", "l_svpt"),
        ("w_1stIn", "l_1stIn"), ("w_1stWon", "l_1stWon"), ("w_2ndWon", "l_2ndWon"),
        ("w_SvGms", "l_SvGms"), ("w_bpSaved", "l_bpSaved"), ("w_bpFaced", "l_bpFaced"),
    ]
    for w_col, l_col in stat_pairs:
        out[w_col] = raw[w_col]
        out[l_col] = raw[l_col]

    # --- Betting Information ---
    out["B365W"] = raw["B365W"]
    out["B365L"] = raw["B365L"]
    out["B365_P1"] = raw["B365W"]
    out["B365_P2"] = raw["B365L"]

    # --- [Engineered] Match Details ---
    tourney_country = raw["Tournament"].map(TOURNAMENT_COUNTRY)
    out["Country"] = tourney_country.map(COUNTRY_CODE)

    # --- [Engineered] Player Information ---
    out["Both RH?"] = ((raw["winner_hand"] == "R") & (raw["loser_hand"] == "R")).astype(int)
    out["Both LH?"] = ((raw["winner_hand"] == "L") & (raw["loser_hand"] == "L")).astype(int)
    out["P1_HF"] = (raw["winner_ioc"] == tourney_country).astype(int)
    out["P2_HF"] = (raw["loser_ioc"] == tourney_country).astype(int)
    out["ht_dif"] = raw["winner_ht"] - raw["loser_ht"]
    out["age_dif"] = raw["winner_age"] - raw["loser_age"]
    out["Pts_dif"] = raw["WPts"] - raw["LPts"]
    out["Rank_dif"] = raw["winner_rank"] - raw["loser_rank"]

    # --- [Engineered] Match Statistics ---
    out["S1_dif"] = raw["W1"] - raw["L1"]
    out["S2_dif"] = raw["W2"] - raw["L2"]
    out["S3_dif"] = raw["W3"] - raw["L3"]
    out["sets_dif"] = raw["Wsets"] - raw["Lsets"]
    out["ace_dif"] = raw["w_ace"] - raw["l_ace"]
    out["df_dif"] = raw["w_df"] - raw["l_df"]
    out["svpt_dif"] = raw["w_svpt"] - raw["l_svpt"]
    out["1stIn_dif"] = raw["w_1stIn"] - raw["l_1stIn"]
    out["1stWon_dif"] = raw["w_1stWon"] - raw["l_1stWon"]
    out["2ndWon_dif"] = raw["w_2ndWon"] - raw["l_2ndWon"]
    out["SvGms_dif"] = raw["w_SvGms"] - raw["l_SvGms"]
    out["bpSaved_dif"] = raw["w_bpSaved"] - raw["l_bpSaved"]
    out["bpFaced_dif"] = raw["w_bpFaced"] - raw["l_bpFaced"]

    # --- [Engineered] Betting Information ---
    out["B365_dif"] = raw["B365W"] - raw["B365L"]
    out["Fav W?"] = (raw["B365W"] < raw["B365L"]).astype(int)

    # Drop rows with no usable rank/points data (can't train on those).
    out = out.dropna(subset=["P1_Rank", "P2_Rank", "P1_Pts", "P2_Pts", "B365_P1", "B365_P2"])

    # Drop rows missing any value the model itself actually consumes as an
    # input feature -- a single NaN anywhere in these columns poisons the
    # entire network's loss (it will train to NaN from epoch 1).
    model_input_cols = [
        "Country", "Surface", "Round", "minutes", "P1_id", "P2_id",
        "P1_hand", "P2_hand", "P1_ht", "P2_ht", "P1_age", "P2_age",
        "Both RH?", "Both LH?", "P1Rank", "P2Rank", "P1Pts", "P2Pts",
        "B365_P1", "B365_P2",
    ]
    before = len(out)
    out = out.dropna(subset=model_input_cols)
    dropped = before - len(out)
    if dropped:
        print(f"Dropped {dropped} rows with missing values in model input columns.")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    # Filter the player dropdown to only currently active players, and
    # enrich with current rank/points, if an active-players file is
    # available. This does NOT affect `out` (the training data above) --
    # retired players stay in the model's history.
    dropdown_players = players.copy()
    dropdown_players["name_normalized"] = dropdown_players["name"].apply(normalize_name)

    if args.active_players and os.path.isfile(args.active_players):
        is_csv = args.active_players.lower().endswith(".csv")

        if is_csv:
            active_df = pd.read_csv(args.active_players)
            active_df["name_normalized"] = active_df["name"].apply(normalize_name)
            for dataset_spelling, rankings_spelling in NAME_ALIASES.items():
                match = active_df[active_df["name_normalized"] == rankings_spelling]
                if not match.empty:
                    alias_row = match.iloc[0].copy()
                    alias_row["name_normalized"] = dataset_spelling
                    active_df = pd.concat([active_df, alias_row.to_frame().T], ignore_index=True)

            dropdown_players = dropdown_players.merge(
                active_df[["name_normalized", "current_rank", "current_pts"]],
                on="name_normalized", how="inner",
            )
        else:
            with open(args.active_players, encoding="utf-8") as f:
                active_raw_names = [line.strip() for line in f if line.strip()]
            active_normalized = {normalize_name(n) for n in active_raw_names}
            for dataset_spelling, rankings_spelling in NAME_ALIASES.items():
                if rankings_spelling in active_normalized:
                    active_normalized.add(dataset_spelling)
            dropdown_players = dropdown_players[
                dropdown_players["name_normalized"].isin(active_normalized)
            ]

        dropdown_players = dropdown_players.drop(columns=["name_normalized"])
        print(
            f"Active-players filter applied: {len(dropdown_players)} of {len(players)} "
            f"players kept for the dropdown (using {args.active_players})."
        )
    else:
        dropdown_players = dropdown_players.drop(columns=["name_normalized"])
        print("No active-players file found -- player dropdown will include all players.")

    players_path = os.path.join(os.path.dirname(args.output), "player_names.csv")
    dropdown_players.to_csv(players_path, index=False)

    print(f"Wrote {len(out)} cleaned rows to: {args.output}")
    print(f"Wrote {len(dropdown_players)} player names to: {players_path}")
    print(f"Columns ({len(out.columns)}): {list(out.columns)}")


if __name__ == "__main__":
    main()
