# import pandas as pd
# import numpy as np
# import joblib
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.config import MODEL_DIR, DATA_DIR, TICKER

# def predict_next_day():
#     model_path = os.path.join(MODEL_DIR, "stock_model.pkl")
#     scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
#     feature_config_path = os.path.join(MODEL_DIR, "feature_config.pkl")
    
#     if not os.path.exists(model_path) or not os.path.exists(scaler_path):
#         print("Model or scaler not found. Train the model first!")
#         return

#     # Load the model, scaler, and feature configuration
#     model = joblib.load(model_path)
#     scaler = joblib.load(scaler_path)
    
#     # Load feature configuration if it exists
#     if os.path.exists(feature_config_path):
#         feature_config = joblib.load(feature_config_path)
#         target_column = feature_config['target_column']
#         feature_columns = feature_config['feature_columns']
#         print(f"Using feature configuration: target={target_column}, features={feature_columns}")
#     else:
#         # Fallback to default column names
#         print("Feature configuration not found. Using default column names.")
#         target_column = 'Close'
#         feature_columns = ['Open', 'High', 'Low', 'Volume']

#     # Load data with the same preprocessing as in training
#     file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    
#     # Skip the header rows as we did during training
#     df = pd.read_csv(file_path, skiprows=2)
    
#     # Rename the first column to 'Date'
#     df = df.rename(columns={df.columns[0]: 'Date'})
    
#     # Convert Date column to datetime
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)
    
#     # Rename the columns to match our feature configuration
#     if not os.path.exists(feature_config_path):
#         # Only rename if we're using default names
#         rename_dict = {}
#         for i, col_name in enumerate(['Close', 'High', 'Low', 'Open', 'Volume']):
#             col_key = f'Unnamed: {i+1}'
#             if col_key in df.columns:
#                 rename_dict[col_key] = col_name
        
#         if rename_dict:
#             df = df.rename(columns=rename_dict)
    
#     # Convert all columns to numeric
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Drop rows with NaN values
#     df = df.dropna()
    
#     print("Data loaded successfully.")
#     print(f"Available columns: {df.columns.tolist()}")
    
#     # Get the latest data
#     latest_data = df.iloc[-1][feature_columns].values.reshape(1, -1)
#     print(f"Latest date in data: {df.index[-1]}")
    
#     # Scale the data and make prediction
#     latest_data_scaled = scaler.transform(latest_data)
#     prediction = model.predict(latest_data_scaled)

#     print(f"Predicted next day's closing price: ${prediction[0]:.2f}")
    
#     # Display the current price for comparison
#     if target_column in df.columns:
#         current_price = df.iloc[-1][target_column]
#         change = prediction[0] - current_price
#         change_percent = (change / current_price) * 100
#         print(f"Current closing price: ${current_price:.2f}")
#         print(f"Predicted change: ${change:.2f} ({change_percent:.2f}%)")

# if __name__ == "__main__":
#     predict_next_day()



import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_DIR, DATA_DIR, TICKER

def predict_next_day(for_streamlit=False):
    model_path = os.path.join(MODEL_DIR, "stock_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    feature_config_path = os.path.join(MODEL_DIR, "feature_config.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Model or scaler not found. Train the model first!")
        return None if for_streamlit else None

    # Load the model, scaler, and feature configuration
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load feature configuration if it exists
    if os.path.exists(feature_config_path):
        feature_config = joblib.load(feature_config_path)
        target_column = feature_config['target_column']
        feature_columns = feature_config['feature_columns']
        print(f"Using feature configuration: target={target_column}, features={feature_columns}")
    else:
        # Fallback to default column names
        print("Feature configuration not found. Using default column names.")
        target_column = 'Close'
        feature_columns = ['Open', 'High', 'Low', 'Volume']

    # Load data with the same preprocessing as in training
    file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    
    # Skip the header rows as we did during training
    df = pd.read_csv(file_path, skiprows=2)
    
    # Rename the first column to 'Date'
    df = df.rename(columns={df.columns[0]: 'Date'})
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Rename the columns to match our feature configuration
    if not os.path.exists(feature_config_path):
        # Only rename if we're using default names
        rename_dict = {}
        for i, col_name in enumerate(['Close', 'High', 'Low', 'Open', 'Volume']):
            col_key = f'Unnamed: {i+1}'
            if col_key in df.columns:
                rename_dict[col_key] = col_name
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    print("Data loaded successfully.")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Get the latest data
    latest_data = df.iloc[-1][feature_columns].values.reshape(1, -1)
    latest_date = df.index[-1]
    print(f"Latest date in data: {latest_date}")
    
    # Scale the data and make prediction
    latest_data_scaled = scaler.transform(latest_data)
    prediction = model.predict(latest_data_scaled)[0]

    print(f"Predicted next day's closing price: ${prediction:.2f}")
    
    # Display the current price for comparison
    if target_column in df.columns:
        current_price = df.iloc[-1][target_column]
        change = prediction - current_price
        change_percent = (change / current_price) * 100
        print(f"Current closing price: ${current_price:.2f}")
        print(f"Predicted change: ${change:.2f} ({change_percent:.2f}%)")
        
        # Return the results if we're in streamlit mode
        if for_streamlit:
            return prediction, current_price, latest_date
    
    return None

if __name__ == "__main__":
    predict_next_day()
