import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from config import DATA_DIR, MODEL_DIR, TEST_SIZE, RANDOM_STATE, N_ESTIMATORS, TICKER

def load_data():
    file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    
    # Read the CSV file, but skip the first 2 rows which contain headers
    # The first 2 rows contain "Ticker", "AAPL", "Date", etc.
    df = pd.read_csv(file_path, skiprows=2)
    
    print("Columns after skipping header rows:", df.columns.tolist())
    print("First few rows after skipping headers:")
    print(df.head())
    
    # The first column appears to contain dates
    # Rename it to 'Date' for clarity
    df = df.rename(columns={df.columns[0]: 'Date'})
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Convert all remaining columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    print("Final processed DataFrame:")
    print(df.head())
    print("DataFrame shape:", df.shape)
    
    return df

def train_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data()
    
    # Check if we have enough data to proceed
    if len(df) < 10:
        print("Error: Not enough data points to train model")
        return
    
    # Use 'Close' as target if it exists, otherwise use the first column
    if 'Close' in df.columns:
        target_col = 'Close'
        feature_cols = [col for col in df.columns if col != 'Close']
    else:
        target_col = df.columns[0]
        feature_cols = df.columns[1:].tolist()
    
    print(f"Using target column: {target_col}")
    print(f"Using feature columns: {feature_cols}")
    
    X = df[feature_cols]
    y = df[target_col]
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training score: {train_score:.4f}")
    print(f"Test score: {test_score:.4f}")

    # Save model & scaler
    joblib.dump(model, os.path.join(MODEL_DIR, "stock_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # Save feature columns to ensure consistency during prediction
    feature_dict = {
        'target_column': target_col,
        'feature_columns': feature_cols
    }
    joblib.dump(feature_dict, os.path.join(MODEL_DIR, "feature_config.pkl"))

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
