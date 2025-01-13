# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:05:39 2024

@author: raucc
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import pickle

def create_improved_model(input_dim, learning_rate=0.001):
    """
    Create an artificial neural network model with:
    - Batch normalization
    - L2 regularization
    """
    model = Sequential([
        # First block
        Dense(256, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second block
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third block
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Fourth block
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Output layer
        Dense(2)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='huber',  # More robust to outliers than mse
        metrics=['mae', 'mse']
    )
    
    return model

def feature_engineering(df):
    """
    Add engineered features to improve model performance
    """
    # Win-loss ratio features
    df['winner_rank_points_ratio'] = df['P1Pts'] / (df['P1Rank'] + 1)
    df['loser_rank_points_ratio'] = df['P2Pts'] / (df['P2Rank'] + 1)
    
    # Age difference
    df['age_difference'] = df['P1_age'] - df['P2_age']
    
    # Height difference
    df['height_difference'] = df['P1_ht'] - df['P2_ht']
    
    # Rank difference
    df['rank_difference'] = df['P2Rank'] - df['P1Rank']
    df['points_difference'] = df['P1Pts'] - df['P2Pts']
    
    # Log transform rank and points
    df['log_P1Rank'] = np.log1p(df['P1Rank'])
    df['log_P2Rank'] = np.log1p(df['P2Rank'])
    df['log_P1Pts'] = np.log1p(df['P1Pts'])
    df['log_P2Pts'] = np.log1p(df['P2Pts'])
    
    return df

def train_model_with_cv(X, y, n_splits=5):
    # Initialize scalers
    input_scaler = MinMaxScaler()  
    output_scaler = MinMaxScaler()
    
    # Scale the data
    X_scaled = input_scaler.fit_transform(X)
    y_scaled = output_scaler.fit_transform(y)
    
    # Initialize K-Fold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store results
    fold_results = []
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]
    
    # Perform K-Fold CV
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        print(f"\nTraining Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]
        
        model = create_improved_model(X_train.shape[1])
        
        # Train with callbacks
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=300,  
            batch_size=64,  
            callbacks=callbacks,
            verbose=1
        )
        
        val_loss = model.evaluate(X_val, y_val)[0]
        fold_results.append(val_loss)
        print(f"Fold {fold + 1} Loss: {val_loss:.4f}")
    
    # Train final model on full dataset
    final_model = create_improved_model(X_scaled.shape[1])
    final_model.fit(
        X_scaled, y_scaled,
        epochs=300,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model and scalers
    final_model.save('tennis_model.h5')
    
    with open('scalers.pkl', 'wb') as f:
        pickle.dump({
            'input_scaler': input_scaler,
            'output_scaler': output_scaler
        }, f)
    
    return np.mean(fold_results), np.std(fold_results)

def main():
    # Load dataset
    dataset_path = r'C:\Users\raucc\OneDrive\Documents\BIA 678 Big Data Technologies\Course Project Part 2\Full Dataset - Cleaned & Feature Engineering.csv'
    df = pd.read_csv(dataset_path)
    
    # Apply feature engineering
    df = feature_engineering(df)
    
    # Updated input features including engineered features
    input_features = [
        "Country", "Surface", "Round", "minutes", "P1_id", 
        "P2_id", "P1_hand", "P2_hand", "P1_ht", 
        "P2_ht", "P1_age", "P2_age", "Both RH?", 
        "Both LH?", "P1Rank", "P2Rank", "P1Pts", "P2Pts",
        # Added engineered features
        "winner_rank_points_ratio", "loser_rank_points_ratio",
        "age_difference", "height_difference",
        "rank_difference", "points_difference",
        "log_P1Rank", "log_P2Rank", "log_P1Pts", "log_P2Pts"
    ]
    
    output_features = ["B365_P1", "B365_P2"]
    
    # Create dictionary to store encoders
    encoders = {}
    
    # Encode categorical variables
    for col in ['Country', 'Surface', 'Round', 'P1_hand', 'P2_hand']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # Save encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save feature lists for the prediction app
    with open('features.pkl', 'wb') as f:
        pickle.dump({
            'input_features': input_features,
            'output_features': output_features
        }, f)
    
    # Prepare data
    X = df[input_features].values
    y = df[output_features].values
    
    # Perform cross-validation and train final model
    mean_loss, std_loss = train_model_with_cv(X, y)
    
    print("\nCross-Validation Results:")
    print(f"Mean Loss: {mean_loss:.4f} (±{std_loss:.4f})")
    print("\nFinal model and scalers have been saved.")

if __name__ == '__main__':
    main()