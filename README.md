# Stock Price Prediction with Stacked LSTM & Sentiment Analysis

This project combines a Stacked LSTM model for stock price prediction with sentiment analysis of stock-related news. The model downloads historical stock data using yfinance, preprocesses the data for a deep learning LSTM model, and forecasts future stock prices. In addition, it fetches news via RSS feeds and performs sentiment analysis using TextBlob to help gauge market sentiment.

## Contributors

- [Navdeep](https://github.com/NavdeepKakrod)
- [Abhishek Kumar](https://github.com/Akabhi2311)
- [Aayush Kumar](https://github.com/Akcodet7)
- [Ashkrit Rai](https://github.com/Askme007)

## Features

- **Historical Stock Data:** Retrieves stock data from Yahoo Finance.
- **Data Preprocessing:** Scales data and transforms it into time series format suitable for LSTM.
- **Stacked LSTM Model:** Uses two LSTM layers and a Dense output layer for prediction.
- **Model Evaluation:** Calculates RMSE on both training and test sets; includes a custom judging score.
- **Future Forecasting:** Predicts the next 30 days of stock prices.
- **News & Sentiment Analysis:**  
  - Fetches stock-related news via RSS feeds.
  - Analyzes headlines using TextBlob to classify sentiment as positive, negative, or neutral.
  - Saves the sentiment analysis results to a CSV file and plots the sentiment distribution.

## Abstract

This project leverages deep learning and natural language processing to predict stock prices and understand market sentiment. By combining a Stacked LSTM network trained on historical closing price data with sentiment analysis of news headlines, the model provides insights that may help in decision-making for stock investments. Although predictions are based on historical data, the integration of sentiment analysis adds an extra dimension to the forecasting process.

## Introduction

Stock market forecasting is a challenging task due to market volatility and a multitude of influencing factors. In this project, we use historical closing prices and recent news sentiment as inputs. The Stacked LSTM model learns long-term dependencies in the data, while sentiment analysis provides an understanding of market mood that might affect stock movements.

## Objective

- **Primary:** Develop a Stacked LSTM model that predicts future stock prices from historical data.
- **Secondary:** Integrate sentiment analysis of stock news to complement the prediction model.

## Methodology

1. **News Fetching & Sentiment Analysis:**  
   - Fetches news headlines via RSS feeds from Yahoo Finance and Seeking Alpha.
   - Analyzes headlines with TextBlob to determine sentiment polarity.
   - Saves the sentiment results to a CSV file for further analysis and visualization.

2. **Stock Data Processing:**  
   - Downloads historical stock data using yfinance.
   - Preprocesses data (scaling, time series creation, train-test split).

3. **Model Building & Training:**  
   - Constructs a Stacked LSTM model with two LSTM layers and one Dense layer.
   - Trains the model on the training data and validates on the test data.
   - Evaluates performance using RMSE and a custom judging score.

4. **Forecasting & Visualization:**  
   - Forecasts stock prices for the next 30 days.
   - Plots the training/test predictions alongside actual stock prices.
   - Displays sentiment distribution from the analyzed news headlines.

## Result Analysis

- **Model Accuracy:**  
  The model computes the Root Mean Squared Error (RMSE) for both training and testing datasets. A lower RMSE indicates that the predictions closely match the actual stock prices.
  
- **Judging Score:**  
  A custom judging score is calculated using the percentage return and variance of the predictions. This score provides an additional perspective on the model's performance by assessing how well the model captures price movement dynamics.
  
- **Visual Insights:**  
  The plotted graphs display:
  - The actual vs. predicted stock prices, highlighting how well the model fits historical data.
  - The training and validation loss trends during model training, demonstrating the convergence of the learning process.
  - Future stock price forecasts over the next 30 days.
  
- **Sentiment Analysis:**  
  The sentiment distribution, visualized via a pie chart, shows the proportions of positive, negative, and neutral news headlines. This information may serve as an additional indicator of market mood and can be used to supplement the technical predictions.

## Tools Used

- **Programming Language:** Python
- **Libraries:**  
  - Data Manipulation: numpy, pandas  
  - Visualization: matplotlib, seaborn  
  - Machine Learning: scikit-learn, TensorFlow, Keras  
  - Data Retrieval: yfinance  
  - News Parsing: feedparser  
  - Sentiment Analysis: TextBlob

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
- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
