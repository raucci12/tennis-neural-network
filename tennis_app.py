# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:10:13 2024

@author: raucc
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pickle
from tensorflow.keras.models import load_model

class TennisPredictionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tennis Match Prediction")
        self.root.geometry("600x800")
        
        # Load the model
        self.model = load_model('tennis_model.h5')
        
        # Load scalers
        with open('scalers.pkl', 'rb') as f:
            scalers = pickle.load(f)
            self.input_scaler = scalers['input_scaler']
            self.output_scaler = scalers['output_scaler']
        
        # Load encoders
        with open('encoders.pkl', 'rb') as f:
            self.encoders = pickle.load(f)
            
        # Load feature lists
        with open('features.pkl', 'rb') as f:
            features = pickle.load(f)
            self.input_features = features['input_features']
            self.output_features = features['output_features']
        
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
        
        # Create input fields
        self.entries = {}
        
        # Add title
        title_label = ttk.Label(self.scrollable_frame, text="Tennis Match Prediction", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=10)
        
        # Categorical inputs
        categorical_fields = ['Country', 'Surface', 'Round', 'winner_hand', 'loser_hand']
        
        # Create sections
        sections = {
            'Match Details': ['Country', 'Surface', 'Round', 'minutes'],
            'Player Details': ['P1_id', 'P2_id', 'P1_hand', 'P2_hand', 
                             'P1_ht', 'P2_ht', 'P1_age', 'P2_age'],
            'Rankings and Points': ['P1Rank', 'P2Rank', 'P1Pts', 'P2Pts'],
            'Additional Info': ['Both RH?', 'Both LH?']
        }
        
        for section_name, fields in sections.items():
            # Add section header
            section_label = ttk.Label(self.scrollable_frame, text=section_name, font=("Helvetica", 12, "bold"))
            section_label.pack(pady=10, padx=5, anchor="w")
            
            # Add separator
            ttk.Separator(self.scrollable_frame, orient='horizontal').pack(fill='x', padx=5, pady=5)
            
            # Create frame for this section
            section_frame = ttk.Frame(self.scrollable_frame)
            section_frame.pack(fill='x', padx=5, pady=5)
            
            for field in fields:
                frame = ttk.Frame(section_frame)
                frame.pack(fill='x', pady=2)
                
                label = ttk.Label(frame, text=field)
                label.pack(side='left')
                
                if field in categorical_fields:
                    combobox = ttk.Combobox(frame, values=list(self.encoders[field].classes_))
                    combobox.pack(side='right')
                    self.entries[field] = combobox
                else:
                    entry = ttk.Entry(frame)
                    entry.pack(side='right')
                    self.entries[field] = entry
        
        # Predict button
        predict_button = ttk.Button(self.scrollable_frame, text="Predict", command=self.predict)
        predict_button.pack(pady=20)
        
        # Results display
        self.result_text = tk.Text(self.scrollable_frame, height=5, width=50)
        self.result_text.pack(pady=10)
    
    def calculate_engineered_features(self, base_features):
        """Calculate engineered features from base inputs"""
        features = base_features.copy()
        
        # Calculate ratios and differences
        features['winner_rank_points_ratio'] = features['P1Pts'] / (features['P1Rank'] + 1)
        features['loser_rank_points_ratio'] = features['P2Pts'] / (features['P2Rank'] + 1)
        features['age_difference'] = features['P1_age'] - features['P2_age']
        features['height_difference'] = features['P1_ht'] - features['P2_ht']
        features['rank_difference'] = features['P2Rank'] - features['P1Rank']
        features['points_difference'] = features['P1Pts'] - features['P2Pts']
        
        # Log transforms
        features['log_P1Rank'] = np.log1p(features['P1Rank'])
        features['log_P2Rank'] = np.log1p(features['P2Rank'])
        features['log_P1Pts'] = np.log1p(features['P1Pts'])
        features['log_P2Pts'] = np.log1p(features['P2Pts'])
        
        return features
    
    def predict(self):
        try:
            # Collect base input values
            base_features = {}
            for field, entry in self.entries.items():
                if field in self.encoders:
                    value = entry.get()
                    index = self.encoders[field].transform([value])[0]
                    base_features[field] = index
                else:
                    base_features[field] = float(entry.get())
            
            # Calculate engineered features
            all_features = self.calculate_engineered_features(base_features)
            
            # Prepare input array in correct order
            input_data = [all_features[feature] for feature in self.input_features]
            
            # Scale input
            input_scaled = self.input_scaler.transform([input_data])
            
            # Make prediction
            prediction_scaled = self.model.predict(input_scaled)
            prediction = self.output_scaler.inverse_transform(prediction_scaled)
            
            # Display results with confidence assessment
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Prediction Results:\n\n")
            self.result_text.insert(tk.END, f"B365_P1 (Win Odds): {prediction[0][0]:.4f}\n")
            self.result_text.insert(tk.END, f"B365_P2 (Loss Odds): {prediction[0][1]:.4f}\n\n")
            
            # Add prediction confidence assessment
            win_loss_ratio = prediction[0][0] / prediction[0][1]
            if win_loss_ratio > 1.5:
                confidence = "High confidence in winner prediction"
            elif win_loss_ratio < 0.67:
                confidence = "High confidence in loser prediction"
            else:
                confidence = "Close match prediction"
            
            self.result_text.insert(tk.END, f"Confidence Assessment: {confidence}")
            
        except ValueError as e:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = TennisPredictionApp()
    app.run()