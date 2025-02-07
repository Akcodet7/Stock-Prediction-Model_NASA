# Stock Price Prediction with Stacked LSTM

A deep learning project that leverages a Stacked LSTM model to predict future stock prices based on historical closing data. The model is built using TensorFlow and Keras, processing data retrieved via yfinance to forecast future trends.

## Contributors

- Navdeep 
- Abhishek Kumar
- Aayush Kumar
- Ashkrit Rai

## Features

- **Historical Data Retrieval:** Downloads stock data (e.g., AAPL) using yfinance.
- **Data Preprocessing:** Extracts closing prices, scales data, and converts it into a time series format.
- **Stacked LSTM Model:** Implements a multi-layer LSTM network to capture temporal dependencies.
- **Performance Evaluation:** Computes RMSE for both training and testing datasets.
- **Future Forecasting:** Predicts the next 30 days of stock prices and visualizes both historical and forecasted data.

## Abstract

This project utilizes a Stacked LSTM network to predict stock prices by analyzing historical closing price data. By preprocessing the data into a time series format and training the model on 65% of the available data, the approach demonstrates promising results in forecasting future trends. Although predictions are based solely on past prices, the model shows considerable accuracy, suggesting that incorporating additional factors (e.g., market sentiment) could further enhance its performance.

## Introduction

Stock market forecasting is challenging due to inherent volatility and multiple influencing factors. This project uses historical closing prices as the primary input for a deep learning model based on a Stacked LSTM architecture. The model is designed to learn long-term dependencies in the data, providing a foundation for making informed predictions about future stock trends.

## Objective

- **Primary Goal:** Develop and evaluate a Stacked LSTM model that accepts historical stock closing prices as input to predict future prices.
- **Secondary Goals:**
  - Retrieve and preprocess data from Yahoo Finance.
  - Create a time series dataset for training and testing.
  - Evaluate model performance using metrics like RMSE.
  - Forecast stock prices for a future period (e.g., next 30 days).

## Methodology

1. **Data Collection:**  
   - Historical stock data is obtained from Yahoo Finance using the yfinance library.
   - Example: Downloading AAPL stock data from January 1, 2015, to May 22, 2024.

2. **Data Preprocessing:**  
   - Extract the closing price from the dataset.
   - Scale the data using MinMaxScaler.
   - Split the dataset into training (65%) and testing (35%) subsets.
   - Transform the linear data into time series format using a fixed time step.

3. **Model Building:**  
   - Construct a Stacked LSTM model with two LSTM layers (the first layer configured with `return_sequences=True`) followed by a Dense layer.
   - Compile the model using the Adam optimizer and mean squared error loss function.

4. **Training & Evaluation:**  
   - Train the model for 100 epochs with a batch size of 64.
   - Evaluate the model using the Root Mean Squared Error (RMSE) on both training and testing datasets.
   - Visualize predictions against actual stock prices.

5. **Future Forecasting:**  
   - Forecast the next 30 days of stock prices.
   - Inverse transform predictions to obtain actual stock prices.
   - Plot historical data alongside future predictions.

## Tools Used

- **Programming Language:** Python
- **Data Collection & Processing:**  
  - [yfinance](https://pypi.org/project/yfinance/) – Retrieve historical stock data.
  - numpy, pandas – Numerical computations and data manipulation.
  - scikit-learn – Data scaling and evaluation.
- **Modeling & Visualization:**  
  - TensorFlow, Keras – Building and training the LSTM model.
  - matplotlib, seaborn – Data visualization.
    
## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Askme007/Stock-Prediction-Model_NASA.git
   cd Stock-Prediction-Model_NASA
   
2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   
3. **Run the Application:**
   - Prediction Script:

     ```bash
     python Stock_Prediction.py

## Result Analysis

- The model achieves competitive RMSE values on both training and testing datasets.
- Visualization plots indicate a close match between predicted and actual stock prices.
- The 30-day future forecast demonstrates the model’s ability to capture stock trends, though accuracy could be improved with additional features such as sentiment analysis.

## Future Scope

- **Incorporating Market Sentiment:**
  - Enhance predictions by integrating NLP-based sentiment analysis from news sources and social media.
- **Model Optimization:**
  - Experiment with additional layers, hyperparameter tuning, and alternative architectures.
- **Extended Forecasting:**
  - Adjust the time step and training strategy to enable longer-term forecasting.

## References

- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time series forecasting](https://towardsdatascience.com/)
- [Time series prediction using deep learning](https://machinelearningmastery.com/)
