# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# from config import DATA_DIR, FIGURES_DIR, TICKER

# def plot_stock_trend():
#     os.makedirs(FIGURES_DIR, exist_ok=True)
    
#     # Read the CSV file, skipping the first 2 rows as done in model_training.py
#     file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
#     df = pd.read_csv(file_path, skiprows=2)
    
#     # The first column appears to contain dates - rename it to 'Date' for clarity
#     df = df.rename(columns={df.columns[0]: 'Date'})
    
#     # Convert Date column to datetime and set as index
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)
    
#     # Convert all remaining columns to numeric
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Drop rows with NaN values
#     df = df.dropna()
    
#     # Use the same target column logic as in model_training.py
#     if 'Close' in df.columns:
#         target_col = 'Close'
#     else:
#         target_col = df.columns[0]  # Use first column as target if 'Close' doesn't exist
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(df[target_col], label=f"{target_col} Price", color="blue")
#     plt.title(f"{TICKER} Stock Price Trend")
#     plt.xlabel("Date")
#     plt.ylabel("Price ($)")
#     plt.legend()
    
#     file_path = os.path.join(FIGURES_DIR, "stock_trend.png")
#     plt.savefig(file_path)
#     plt.show()
#     print(f"Plot saved at {file_path}")

# if __name__ == "__main__":
#     plot_stock_trend()



# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.config import DATA_DIR, FIGURES_DIR, TICKER

# def plot_stock_trend(for_streamlit=False):
#     os.makedirs(FIGURES_DIR, exist_ok=True)
    
#     # Read the CSV file, skipping the first 2 rows as done in model_training.py
#     file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
#     df = pd.read_csv(file_path, skiprows=2)
    
#     # The first column appears to contain dates - rename it to 'Date' for clarity
#     df = df.rename(columns={df.columns[0]: 'Date'})
    
#     # Convert Date column to datetime and set as index
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)
    
#     # Convert all remaining columns to numeric
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
    
#     # Drop rows with NaN values
#     df = df.dropna()
    
#     # Use the same target column logic as in model_training.py
#     if 'Close' in df.columns:
#         target_col = 'Close'
#     else:
#         target_col = df.columns[0]  # Use first column as target if 'Close' doesn't exist
    
#     # Create figure
#     fig, ax = plt.subplots(figsize=(12, 6))
    
#     # Plot last 90 days of data to make it more readable
#     days_to_plot = 90
#     recent_data = df.iloc[-days_to_plot:] if len(df) > days_to_plot else df
    
#     # Plot closing price
#     recent_data[target_col].plot(ax=ax, color="blue", label=f"{target_col} Price")
    
#     # Add moving averages
#     recent_data[target_col].rolling(window=20).mean().plot(
#         ax=ax, color="red", label="20-Day MA"
#     )
#     recent_data[target_col].rolling(window=50).mean().plot(
#         ax=ax, color="green", label="50-Day MA"
#     )
    
#     ax.set_title(f"{TICKER} Stock Price Trend")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Price ($)")
#     ax.legend()
#     ax.grid(True)
    
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
#     plt.tight_layout()
    
#     if for_streamlit:
#         return fig
#     else:
#         file_path = os.path.join(FIGURES_DIR, "stock_trend.png")
#         plt.savefig(file_path)
#         plt.show()
#         print(f"Plot saved at {file_path}")
#         return None

# if __name__ == "__main__":
#     plot_stock_trend()


import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, FIGURES_DIR, TICKER

def plot_stock_trend(for_streamlit=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Read the CSV file, skipping the first 2 rows as done in model_training.py
    file_path = os.path.join(DATA_DIR, f"{TICKER}.csv")
    df = pd.read_csv(file_path, skiprows=2)
    
    # The first column appears to contain dates - rename it to 'Date' for clarity
    df = df.rename(columns={df.columns[0]: 'Date'})
    
    # Convert Date column to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Convert all remaining columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Use the same target column logic as in model_training.py
    if 'Close' in df.columns:
        target_col = 'Close'
    else:
        target_col = df.columns[0]  # Use first column as target if 'Close' doesn't exist
    
    # Get volume column if available
    volume_col = None
    for possible_name in ['Volume', 'volume', 'vol', 'Unnamed: 5']:
        if possible_name in df.columns:
            volume_col = possible_name
            break
    
    # Plot last 180 days of data to make it more readable but include enough history
    days_to_plot = 180
    recent_data = df.iloc[-days_to_plot:] if len(df) > days_to_plot else df
    
    # Calculate moving averages - these are helpful indicators in stock analysis
    recent_data['MA20'] = recent_data[target_col].rolling(window=20).mean()  # 20-day moving average (short term trend)
    recent_data['MA50'] = recent_data[target_col].rolling(window=50).mean()  # 50-day moving average (medium term trend)
    recent_data['MA200'] = recent_data[target_col].rolling(window=200).mean()  # 200-day moving average (long term trend)
    
    # Create subplots with gridspec for better control
    if volume_col:
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = None
    
    # Style the price plot
    ax1.set_facecolor('#f8f9fa')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot price and moving averages
    ax1.plot(recent_data.index, recent_data[target_col], linewidth=2, label=f'{target_col}', color='#2c3e50')
    ax1.plot(recent_data.index, recent_data['MA20'], linewidth=1.5, label='20-Day MA (Short Term)', color='#e74c3c')
    ax1.plot(recent_data.index, recent_data['MA50'], linewidth=1.5, label='50-Day MA (Medium Term)', color='#2ecc71')
    
    # Only plot 200-day MA if we have enough data
    if len(recent_data) > 200:
        ax1.plot(recent_data.index, recent_data['MA200'], linewidth=1.5, label='200-Day MA (Long Term)', color='#3498db')
    
    # Highlight the most recent price point
    latest_date = recent_data.index[-1]
    latest_price = recent_data[target_col].iloc[-1]
    ax1.scatter(latest_date, latest_price, color='red', s=100, zorder=5)
    ax1.annotate(f'${latest_price:.2f}', 
                 xy=(latest_date, latest_price),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontweight='bold',
                 color='red')
    
    # Add volume subplot if available
    if volume_col and ax2:
        # Plot volume as bar chart
        ax2.bar(recent_data.index, recent_data[volume_col], width=1, label='Volume', color='#7f8c8d', alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_facecolor('#f8f9fa')
        
        # Format y-axis to show volume in millions/billions
        ax2.ticklabel_format(style='plain', axis='y')
        
        # Only show every 30 days on x-axis to avoid crowding
        ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    
    # Style the main price chart
    ax1.set_title(f'{TICKER} Stock Price and Technical Indicators', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='upper left')
    
    # Set custom date formatting
    date_formatter = DateFormatter('%b %d')
    ax1.xaxis.set_major_formatter(date_formatter)
    plt.xticks(rotation=45)
    
    # Add some market regime analysis
    try:
        # Determine if we're in an uptrend, downtrend, or sideways market
        short_term = recent_data['MA20'].iloc[-1] > recent_data['MA20'].iloc[-30]
        medium_term = recent_data['MA50'].iloc[-1] > recent_data['MA50'].iloc[-30]
        
        trend_text = ""
        if short_term and medium_term:
            trend_text = "UPTREND: Short and medium-term indicators suggest bullish momentum"
            trend_color = "green"
        elif not short_term and not medium_term:
            trend_text = "DOWNTREND: Short and medium-term indicators suggest bearish momentum"
            trend_color = "red"
        else:
            trend_text = "SIDEWAYS: Mixed signals in short and medium-term indicators"
            trend_color = "orange"
        
        # Add trend annotation
        plt.figtext(0.5, 0.01, trend_text, ha='center', color=trend_color, fontsize=12, fontweight='bold')
    except:
        # Skip if we can't calculate the trend
        pass
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if for_streamlit:
        return fig
    else:
        file_path = os.path.join(FIGURES_DIR, f"{TICKER}_technical_analysis.png")
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plot saved at {file_path}")
        return None

if __name__ == "__main__":
    plot_stock_trend()