# import streamlit as st
# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from src.prediction import predict_next_day

# st.title("Stock Price Prediction App")

# if st.button("Predict Next Day's Price"):
#     predict_next_day()



import streamlit as st
import sys
import os
import pandas as pd
import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules from your project
from src.prediction import predict_next_day
from src.visualization import plot_stock_trend
from src.config import TICKER

st.title("Stock Price Prediction App")

if st.button("Predict Next Day's Price"):
    # Create columns for organizing the output
    col1, col2 = st.columns(2)
    
    # Get prediction results
    with st.spinner("Making prediction..."):
        prediction_results = predict_next_day(for_streamlit=True)
        
        if prediction_results:
            prediction, current_price, latest_date = prediction_results
            
            # Calculate the next business day (rough approximation)
            next_date = latest_date + datetime.timedelta(days=1)
            if next_date.weekday() >= 5:  # Saturday or Sunday
                next_date += datetime.timedelta(days=(7 - next_date.weekday()))
                
            # Display the prediction results
            with col1:
                st.subheader("Prediction Results")
                st.metric(
                    label=f"Predicted Price ({next_date.strftime('%Y-%m-%d')})",
                    value=f"${prediction:.2f}",
                    delta=f"{((prediction - current_price) / current_price) * 100:.2f}%"
                )
                st.metric(
                    label=f"Current Price ({latest_date.strftime('%Y-%m-%d')})",
                    value=f"${current_price:.2f}"
                )
                
                # Add explanatory text
                change = prediction - current_price
                direction = "increase" if change > 0 else "decrease"
                st.write(f"The model predicts a ${abs(change):.2f} {direction} in stock price for the next trading day.")
    
    # Show the visualization
    with st.spinner("Generating visualization..."):
        with col2:
            st.subheader("Historical Price Trend")
            fig = plot_stock_trend(for_streamlit=True)
            if fig:
                st.pyplot(fig)
            else:
                st.error("Could not generate visualization.")
                
    # Add some additional information
    st.info(f"This prediction is based on historical price data for {TICKER} stock.")

# Add a sidebar with additional information
with st.sidebar:
    st.subheader("About")
    st.write(f"This app predicts the next day's closing price for {TICKER} stock.")
    st.write("The model is trained on historical price data.")
