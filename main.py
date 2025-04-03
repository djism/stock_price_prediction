# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# import data_preprocessing
# import model_training
# import prediction
# import visualization

# def main():
#     print("Fetching stock data...")
#     data_preprocessing.fetch_stock_data()

#     print("Training model...")
#     model_training.train_model()

#     print("Making predictions...")
#     prediction.predict_next_day()

#     print("Generating visualizations...")
#     visualization.plot_stock_trend()

# if __name__ == "__main__":
#     main()


import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import data_preprocessing
import model_training
import prediction
import visualization

def main():
    print("Fetching stock data...")
    data_preprocessing.fetch_stock_data()

    print("Training model...")
    model_training.train_model()

    print("Making predictions...")
    prediction.predict_next_day()

    print("Generating visualizations...")
    visualization.plot_stock_trend()

if __name__ == "__main__":
    main()