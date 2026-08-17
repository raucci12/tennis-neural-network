# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:10:13 2024
Updated to use human-readable dropdowns for court surface, country, hand,
and player names -- with automatic height/age lookup for known players
(projected to the 2025 Grand Slam selected), and manual entry for an
"Unknown Player" option.

@author: raucc
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNKNOWN_PLAYER_LABEL = "-- Unknown Player (enter manually) --"

# Human-readable name -> underlying code used during training.
# These must match the encoding done in clean_data.py / tennis_cv.py.
COUNTRY_NAME_TO_CODE = {
    "Australia": "1",
    "France": "2",
    "Great Britain": "3",
    "USA": "4",
}
SURFACE_NAME_TO_CODE = {
    "Hard": "1",
    "Clay": "2",
    "Grass": "3",
}
HAND_NAME_TO_CODE = {
    "Right-handed": "1.0",
    "Left-handed": "2.0",
}
CODE_TO_HAND_NAME = {v: k for k, v in HAND_NAME_TO_CODE.items()}

# Each WTA Grand Slam is always played on the same surface, so once the
# tournament country is chosen, the surface isn't really a separate choice.
COUNTRY_TO_SURFACE = {
    "Australia": "Hard",
    "France": "Clay",
    "Great Britain": "Grass",
    "USA": "Hard",
}

# Slider bounds for the match-duration fine-tuning control, based on the
# actual range of match lengths in the training data (~41-232 minutes).
MINUTES_MIN = 40
MINUTES_MAX = 230
MINUTES_DEFAULT = 90

# Nicer on-screen labels for fields whose internal key is a code name.
DISPLAY_LABELS = {
    "Country": "Tournament Country",
    "Surface": "Court Surface",
    "Round": "Round",
    "minutes": "Match Duration (minutes)",
    "P1_id": "Player 1",
    "P2_id": "Player 2",
    "P1_hand": "Player 1 Dominant Hand",
    "P2_hand": "Player 2 Dominant Hand",
    "P1_ht": "Player 1 Height (cm)",
    "P2_ht": "Player 2 Height (cm)",
    "P1_age": "Player 1 Age",
    "P2_age": "Player 2 Age",
    "P1Rank": "Player 1 WTA Rank",
    "P2Rank": "Player 2 WTA Rank",
    "P1Pts": "Player 1 WTA Points",
    "P2Pts": "Player 2 WTA Points",
    "Both RH?": "Both Players Right-Handed?",
    "Both LH?": "Both Players Left-Handed?",
}


class TennisPredictionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tennis Match Prediction")
        self.root.geometry("700x950")

        # Load the model
        self.model = load_model(os.path.join(SCRIPT_DIR, "tennis_model.h5"))

        # Load scalers
        with open(os.path.join(SCRIPT_DIR, "scalers.pkl"), "rb") as f:
            scalers = pickle.load(f)
            self.input_scaler = scalers["input_scaler"]
            self.output_scaler = scalers["output_scaler"]

        # Load encoders
        with open(os.path.join(SCRIPT_DIR, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)

        # Load feature lists
        with open(os.path.join(SCRIPT_DIR, "features.pkl"), "rb") as f:
            features = pickle.load(f)
            self.input_features = features["input_features"]
            self.output_features = features["output_features"]

        # Load player names + height + projected 2025 ages for the
        # P1_id / P2_id dropdowns.
        self.player_data = {}  # display name -> {"id", "height", "age_by_country", "rank", "pts"}
        players_path = os.path.join(SCRIPT_DIR, "data", "player_names.csv")
        if os.path.isfile(players_path):
            players_df = pd.read_csv(players_path)
            for _, row in players_df.iterrows():
                display = f"{row['name']} (ID {int(row['player_id'])})"
                self.player_data[display] = {
                    "id": row["player_id"],
                    "height": row["height"] if pd.notna(row["height"]) else None,
                    "hand": row["hand"] if pd.notna(row["hand"]) else None,
                    "age_by_country": {
                        "1": row.get("age_country_1"),
                        "2": row.get("age_country_2"),
                        "3": row.get("age_country_3"),
                        "4": row.get("age_country_4"),
                    },
                    "rank": row["current_rank"] if "current_rank" in row and pd.notna(row["current_rank"]) else None,
                    "pts": row["current_pts"] if "current_pts" in row and pd.notna(row["current_pts"]) else None,
                }
        self.player_display_names = [UNKNOWN_PLAYER_LABEL] + sorted(self.player_data.keys())

        # Cache of the last successfully validated inputs, so the duration
        # slider can re-run predictions live without re-validating every
        # other field on each drag.
        self.last_base_features = None
        self.last_player_labels = None  # (p1_name, p2_name) for the results display

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=1)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Enable mouse-wheel scrolling. Windows/Mac send <MouseWheel> with a
        # signed event.delta; Linux sends separate <Button-4>/<Button-5>
        # events instead, so both are handled here for portability.
        def _on_mousewheel_windows(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux(event):
            canvas.yview_scroll(-1 if event.num == 4 else 1, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel_windows)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        # Only scroll with the wheel while the cursor is actually over the
        # form -- binding it globally with no unbind would also hijack
        # scrolling on other windows/widgets.
        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        # Create input fields
        self.entries = {}

        # Add title
        title_label = ttk.Label(self.scrollable_frame, text="Tennis Match Prediction", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=10)
        subtitle_label = ttk.Label(
            self.scrollable_frame,
            text="Selecting a known player auto-fills height, hand, age, rank, and points.\nChoosing a Tournament Country auto-fills the Court Surface. Choose \"Unknown Player\" to enter player details manually.",
            font=("Helvetica", 9, "italic"),
            justify="center",
        )
        subtitle_label.pack(pady=(0, 10))

        player_fields = ["P1_id", "P2_id"]
        country_surface_fields = ["Country", "Surface"]
        hand_fields = ["P1_hand", "P2_hand"]
        encoder_only_fields = ["Round"]  # already stored as real text, e.g. "1st Round"
        auto_fill_fields = ["P1_ht", "P2_ht", "P1_age", "P2_age"]

        sections = {
            "Match Details": ["Country", "Surface", "Round"],
            "Player Details": ["P1_id", "P2_id", "P1_hand", "P2_hand",
                                "P1_ht", "P2_ht", "P1_age", "P2_age"],
            "Rankings and Points": ["P1Rank", "P2Rank", "P1Pts", "P2Pts"],
        }

        for section_name, fields in sections.items():
            section_label = ttk.Label(self.scrollable_frame, text=section_name, font=("Helvetica", 12, "bold"))
            section_label.pack(pady=10, padx=5, anchor="w")

            ttk.Separator(self.scrollable_frame, orient="horizontal").pack(fill="x", padx=5, pady=5)

            section_frame = ttk.Frame(self.scrollable_frame)
            section_frame.pack(fill="x", padx=5, pady=5)

            for field in fields:
                frame = ttk.Frame(section_frame)
                frame.pack(fill="x", pady=2)

                label_text = DISPLAY_LABELS.get(field, field)
                label = ttk.Label(frame, text=label_text)
                label.pack(side="left")

                if field in player_fields:
                    combobox = ttk.Combobox(frame, values=self.player_display_names, width=32, state="readonly")
                    combobox.pack(side="right")
                    combobox.bind("<<ComboboxSelected>>", lambda e, f=field: self.on_player_selected(f))
                    self.entries[field] = combobox

                elif field in country_surface_fields:
                    name_map = COUNTRY_NAME_TO_CODE if field == "Country" else SURFACE_NAME_TO_CODE
                    combobox = ttk.Combobox(frame, values=list(name_map.keys()), width=32, state="readonly")
                    combobox.pack(side="right")
                    if field == "Country":
                        combobox.bind("<<ComboboxSelected>>", lambda e: self.on_country_selected())
                    self.entries[field] = combobox

                elif field in hand_fields:
                    combobox = ttk.Combobox(frame, values=list(HAND_NAME_TO_CODE.keys()), width=32, state="readonly")
                    combobox.pack(side="right")
                    self.entries[field] = combobox

                elif field in encoder_only_fields:
                    combobox = ttk.Combobox(frame, values=list(self.encoders[field].classes_), width=32, state="readonly")
                    combobox.pack(side="right")
                    self.entries[field] = combobox

                else:
                    entry = ttk.Entry(frame)
                    entry.pack(side="right")
                    self.entries[field] = entry

        # Predict button
        predict_button = ttk.Button(self.scrollable_frame, text="Predict", command=self.predict)
        predict_button.pack(pady=20)

        # Fine-tuning slider: after an initial prediction, drag this to see
        # how the odds change for a longer or shorter match, without
        # re-entering every other field.
        duration_frame = ttk.Frame(self.scrollable_frame)
        duration_frame.pack(fill="x", padx=5, pady=(0, 10))

        duration_header = ttk.Label(duration_frame, text="Fine-Tune Match Duration", font=("Helvetica", 12, "bold"))
        duration_header.pack(anchor="w")
        duration_hint = ttk.Label(
            duration_frame,
            text="Predict once first, then drag to see how odds change for a longer or shorter match.",
            font=("Helvetica", 9, "italic"),
        )
        duration_hint.pack(anchor="w", pady=(0, 5))

        self.minutes_value_label = ttk.Label(
            duration_frame, text=f"{MINUTES_DEFAULT} minutes", font=("Helvetica", 10, "bold")
        )
        self.minutes_value_label.pack(anchor="w")

        self.minutes_slider = ttk.Scale(
            duration_frame, from_=MINUTES_MIN, to=MINUTES_MAX,
            orient="horizontal", command=self.on_minutes_change,
        )
        self.minutes_slider.set(MINUTES_DEFAULT)
        self.minutes_slider.pack(fill="x")
        self.entries["minutes"] = self.minutes_slider

        # Results display
        self.result_text = tk.Text(self.scrollable_frame, height=14, width=70, wrap="word")
        self.result_text.pack(pady=(10, 30))

    def _player_field_num(self, field):
        """'P1_id' -> '1', 'P2_id' -> '2'"""
        return "1" if field == "P1_id" else "2"

    def on_player_selected(self, field):
        """Called when the user picks a name in the P1_id / P2_id dropdown.
        Auto-fills (and locks) height/hand/age/rank/points for a known
        player; unlocks manual entry for the Unknown Player option."""
        num = self._player_field_num(field)
        ht_field = f"P{num}_ht"
        age_field = f"P{num}_age"
        hand_field = f"P{num}_hand"
        rank_field = f"P{num}Rank"
        pts_field = f"P{num}Pts"
        selection = self.entries[field].get()

        if selection == UNKNOWN_PLAYER_LABEL or selection not in self.player_data:
            # Unlock manual entry
            for f in (ht_field, age_field, rank_field, pts_field):
                entry = self.entries[f]
                entry.configure(state="normal")
                entry.delete(0, "end")
            hand_combo = self.entries[hand_field]
            hand_combo.configure(state="readonly")
            hand_combo.set("")
            return

        info = self.player_data[selection]

        ht_entry = self.entries[ht_field]
        ht_entry.configure(state="normal")
        ht_entry.delete(0, "end")
        if info["height"] is not None:
            ht_entry.insert(0, f"{info['height']:.1f}")
            ht_entry.configure(state="disabled")
        else:
            # No recorded height for this player -- leave it editable so
            # the user can supply one.
            pass

        rank_entry = self.entries[rank_field]
        rank_entry.configure(state="normal")
        rank_entry.delete(0, "end")
        if info["rank"] is not None:
            rank_entry.insert(0, f"{int(info['rank'])}")
            rank_entry.configure(state="disabled")

        pts_entry = self.entries[pts_field]
        pts_entry.configure(state="normal")
        pts_entry.delete(0, "end")
        if info["pts"] is not None:
            pts_entry.insert(0, f"{int(info['pts'])}")
            pts_entry.configure(state="disabled")

        hand_combo = self.entries[hand_field]
        if info["hand"] is not None:
            hand_code_str = f"{info['hand']:.1f}"  # matches encoder classes, e.g. "1.0"
            hand_name = CODE_TO_HAND_NAME.get(hand_code_str)
            if hand_name is not None:
                hand_combo.configure(state="readonly")
                hand_combo.set(hand_name)
                hand_combo.configure(state="disabled")
        else:
            # No recorded hand for this player -- leave it selectable so
            # the user can supply one.
            hand_combo.configure(state="readonly")
            hand_combo.set("")

        self._fill_age_for_player(field)

    def _fill_age_for_player(self, field):
        """Fill (and lock) the age field for whichever known player is
        selected in `field`, based on the currently selected tournament."""
        num = self._player_field_num(field)
        age_field = f"P{num}_age"
        selection = self.entries[field].get()

        if selection == UNKNOWN_PLAYER_LABEL or selection not in self.player_data:
            return

        country_name = self.entries["Country"].get()
        country_code = COUNTRY_NAME_TO_CODE.get(country_name)

        age_entry = self.entries[age_field]
        age_entry.configure(state="normal")
        age_entry.delete(0, "end")

        if country_code is None:
            # No tournament selected yet -- can't project age until they pick one.
            return

        age_value = self.player_data[selection]["age_by_country"].get(country_code)
        if age_value is not None and not pd.isna(age_value):
            age_entry.insert(0, f"{age_value:.1f}")
            age_entry.configure(state="disabled")

    def on_country_selected(self):
        """When the tournament changes: auto-fill (and lock) the Surface,
        since each Grand Slam is always played on the same surface -- and
        refresh the age for any known player currently selected in P1/P2
        (age is tournament-specific)."""
        country_name = self.entries["Country"].get()
        surface_name = COUNTRY_TO_SURFACE.get(country_name)
        if surface_name is not None:
            surface_combo = self.entries["Surface"]
            surface_combo.configure(state="readonly")
            surface_combo.set(surface_name)
            surface_combo.configure(state="disabled")

        for field in ("P1_id", "P2_id"):
            self._fill_age_for_player(field)

    def calculate_engineered_features(self, base_features):
        """Calculate engineered features from base inputs"""
        features = base_features.copy()

        features["winner_rank_points_ratio"] = features["P1Pts"] / (features["P1Rank"] + 1)
        features["loser_rank_points_ratio"] = features["P2Pts"] / (features["P2Rank"] + 1)
        features["age_difference"] = features["P1_age"] - features["P2_age"]
        features["height_difference"] = features["P1_ht"] - features["P2_ht"]
        features["rank_difference"] = features["P2Rank"] - features["P1Rank"]
        features["points_difference"] = features["P1Pts"] - features["P2Pts"]

        features["log_P1Rank"] = np.log1p(features["P1Rank"])
        features["log_P2Rank"] = np.log1p(features["P2Rank"])
        features["log_P1Pts"] = np.log1p(features["P1Pts"])
        features["log_P2Pts"] = np.log1p(features["P2Pts"])

        return features

    def predict(self):
        try:
            base_features = {}
            hand_codes = {}  # raw "1.0"/"2.0" codes, kept aside to derive Both RH?/LH?
            for field, entry in self.entries.items():
                value = entry.get()

                if field in ("P1_id", "P2_id"):
                    if value == UNKNOWN_PLAYER_LABEL:
                        base_features[field] = 0.0  # placeholder ID for an unknown player
                    elif value in self.player_data:
                        base_features[field] = float(self.player_data[value]["id"])
                    else:
                        raise ValueError(f"Please select a player for {DISPLAY_LABELS.get(field, field)}.")

                elif field == "Country":
                    code = COUNTRY_NAME_TO_CODE.get(value)
                    if code is None:
                        raise ValueError("Please select a Tournament Country.")
                    base_features[field] = self.encoders["Country"].transform([code])[0]

                elif field == "Surface":
                    code = SURFACE_NAME_TO_CODE.get(value)
                    if code is None:
                        raise ValueError("Please select a Court Surface.")
                    base_features[field] = self.encoders["Surface"].transform([code])[0]

                elif field in ("P1_hand", "P2_hand"):
                    code = HAND_NAME_TO_CODE.get(value)
                    if code is None:
                        raise ValueError(f"Please select {DISPLAY_LABELS.get(field, field)}.")
                    hand_codes[field] = code
                    base_features[field] = self.encoders[field].transform([code])[0]

                elif field in self.encoders:  # Round
                    base_features[field] = self.encoders[field].transform([value])[0]

                elif field in ("P1_ht", "P2_ht", "P1_age", "P2_age"):
                    if not value.strip():
                        raise ValueError(
                            f"{DISPLAY_LABELS.get(field, field)} is required "
                            "(auto-fills for a known player, or enter manually for Unknown Player)."
                        )
                    base_features[field] = float(value)

                else:
                    base_features[field] = float(value)

            # Derived automatically from the two hand selections above --
            # no separate input needed from the user.
            p1_hand_code = hand_codes.get("P1_hand")
            p2_hand_code = hand_codes.get("P2_hand")
            base_features["Both RH?"] = int(p1_hand_code == "1.0" and p2_hand_code == "1.0")
            base_features["Both LH?"] = int(p1_hand_code == "2.0" and p2_hand_code == "2.0")

            self.last_base_features = dict(base_features)
            self.last_player_labels = (
                self._clean_player_label(self.entries["P1_id"].get()) or "Player 1",
                self._clean_player_label(self.entries["P2_id"].get()) or "Player 2",
            )
            self._run_model(base_features)

        except ValueError as e:
            messagebox.showerror("Input Error", str(e) if str(e) else "Please enter valid values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _clean_player_label(self, display_value):
        """'Aryna Sabalenka (ID 214544)' -> 'Aryna Sabalenka'. Returns None
        for the Unknown Player option or an empty selection."""
        if not display_value or display_value == UNKNOWN_PLAYER_LABEL:
            return None
        return display_value.split(" (ID ")[0]

    def _run_model(self, base_features):
        """Scale, predict, and display results for a fully-populated
        base_features dict. Shared by the Predict button and the live
        duration slider."""
        all_features = self.calculate_engineered_features(base_features)
        input_data = [all_features[feature] for feature in self.input_features]
        input_scaled = self.input_scaler.transform([input_data])

        prediction_scaled = self.model.predict(input_scaled, verbose=0)
        prediction = self.output_scaler.inverse_transform(prediction_scaled)
        p1_odds, p2_odds = prediction[0][0], prediction[0][1]

        p1_name, p2_name = self.last_player_labels or ("Player 1", "Player 2")

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Prediction Results:\n\n")
        self.result_text.insert(tk.END, f"B365_P1 (Win Odds): {p1_odds:.4f} -- {p1_name}\n")
        self.result_text.insert(tk.END, f"B365_P2 (Loss Odds): {p2_odds:.4f} -- {p2_name}\n\n")

        # Lower decimal odds = higher implied win probability = the favorite.
        if p1_odds < p2_odds:
            favorite_name = p1_name
        elif p2_odds < p1_odds:
            favorite_name = p2_name
        else:
            favorite_name = None

        if favorite_name is not None:
            self.result_text.insert(tk.END, f"Favorite: {favorite_name}\n\n")
        else:
            self.result_text.insert(tk.END, "Favorite: Too close to call\n\n")

        win_loss_ratio = p1_odds / p2_odds
        if win_loss_ratio > 1.5:
            confidence = "High confidence in winner prediction"
        elif win_loss_ratio < 0.67:
            confidence = "High confidence in loser prediction"
        else:
            confidence = "Close match prediction"

        self.result_text.insert(tk.END, f"Confidence Assessment: {confidence}")

    def on_minutes_change(self, value):
        """Called continuously as the duration slider is dragged. Updates
        the on-screen minutes label always; only re-runs the model live if
        a first prediction has already been made (so we have a full,
        already-validated set of inputs to reuse)."""
        try:
            minutes_val = float(value)
        except (TypeError, ValueError):
            return

        self.minutes_value_label.configure(text=f"{minutes_val:.0f} minutes")

        if self.last_base_features is not None:
            updated_features = dict(self.last_base_features)
            updated_features["minutes"] = minutes_val
            self._run_model(updated_features)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TennisPredictionApp()
    app.run()
