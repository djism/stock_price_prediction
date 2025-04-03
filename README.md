# Stock Market Price Prediction


A machine learning application that predicts stock prices using historical data and visualizes market trends with technical indicators.


Project Overview:- 
This project implements a stock price prediction system using Random Forest regression models to forecast next-day stock prices. The application includes data preprocessing, model training, prediction capabilities, and technical analysis visualizations to assist with investment decisions.
The system is designed to:

Fetch historical stock data for any ticker symbol (default: AAPL)
Preprocess and clean the data for machine learning
Train a Random Forest regression model on the historical data
Generate predictions for the next trading day's closing price
Visualize stock price trends with technical indicators
Present results through a Streamlit web application

Features:-

Data Collection & Preprocessing

Automatic data fetching from Yahoo Finance using the yfinance library
Handling of date formatting and missing values
Scaling of feature data for optimal model performance

Model Training

Implementation of Random Forest Regressor for price prediction
Train/test split for model validation
Model persistence using joblib for future predictions
Feature importance analysis

Technical Analysis

Moving average calculations (20-day, 50-day, and 200-day)
Market trend identification (uptrend, downtrend, or sideways)
Volume analysis
Price momentum indicators

Visualization

Interactive price charts with trend lines
Volume analysis subplot
Technical indicator overlays
Highlighted current price and predicted price

Web Application

User-friendly Streamlit interface
One-click prediction generation
Clear presentation of prediction results
Visual comparison of current vs. predicted prices

Technical Architecture
The project is organized into several modules:



![Alt text](https://github.com/djism/stock_price_prediction/blob/main/Screenshot%202025-04-03%20045656.png?raw=true)



Technical Implementation Details:- 

Data Pipeline
The data pipeline fetches historical stock data and processes it for machine learning:

Download data using Yahoo Finance API
Clean and normalize data for machine learning
Feature engineering for improved model performance
Scale features using MinMaxScaler

Machine Learning Model
The project uses Random Forest Regression with these key features:

Algorithm: Random Forest Regressor
Features: Open, High, Low, Volume (and derived indicators)
Target: Next day's closing price
Hyperparameters: 100 estimators, random state for reproducibility
Evaluation: R² score on training and test sets

Visualization Engine
The visualization component creates comprehensive technical analysis charts:

Price history with moving averages
Volume analysis
Market trend identification
Price momentum visualization
Annotated prediction points


Web Application
The Streamlit application provides a user-friendly interface for interacting with the prediction model:


![Alt text](https://github.com/djism/stock_price_prediction/blob/main/Screenshot%202025-04-03%20044014.png?raw=true)

![Alt text](https://github.com/djism/stock_price_prediction/blob/main/Screenshot%202025-04-03%20044025.png?raw=true)


Features include:

One-click prediction generation
Clear presentation of prediction results
Visual price trend analysis
Technical indicator explanation

Installation & Usage:-
Prerequisites

-Python 3.7+
-pip package manager

Setup

-Clone the repository

git clone https://github.com/yourusername/stock-prediction.git

cd stock-prediction

-Install required packages

pip install -r requirements.txt

-Run the main application

python main.py

-Launch the web application

streamlit run app.py




Configuration
You can modify the config.py file to:

Change the stock ticker symbol (default: AAPL)
Adjust date ranges for historical data
Modify model hyperparameters
Configure paths for data storage

Future Improvements
Potential enhancements for future versions:

Support for multiple stock symbols
Additional machine learning models (LSTM, XGBoost)
Sentiment analysis integration from news and social media
Portfolio optimization recommendations
Backtesting capabilities to evaluate model performance
Advanced technical indicators
Risk assessment metrics

Technologies Used

Python: Core programming language
pandas: Data manipulation and analysis
scikit-learn: Machine learning algorithms
yfinance: Stock data acquisition
matplotlib/seaborn: Visualization
Streamlit: Web application framework
joblib: Model persistence
TensorFlow: (Prepared for future deep learning models)

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Yahoo Finance for providing access to historical stock data
The scikit-learn team for their excellent machine learning library
Streamlit for their intuitive web application framework
