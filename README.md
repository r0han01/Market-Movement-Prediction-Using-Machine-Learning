# Market Movement Prediction Using Machine Learning

## Overview
This project aims to predict the movement of the Dow Jones Industrial Average (DJI) index using machine learning techniques. The model leverages features such as retail sales changes, sentiment analysis, technical indicators, and economic factors. A combination of different machine learning models, including Bi-directional GRU (BGRU), neural networks, and ensemble methods, were evaluated for their effectiveness in capturing patterns from the dataset.

### Project Goal
The goal is to predict the market movement (positive or negative) based on historical data and features like sentiment, retail sales, DJI index changes, and other indicators. This can help in forecasting market behavior to assist investors and analysts in making more informed decisions.

## Key Features
- **Retail Sales Change**: Percentage change in retail sales.
- **DJI 7-Day % Change**: Percentage change in the Dow Jones Industrial Average over the past 7 days.
- **Sentiment**: Market sentiment (positive/negative).
- **Technical Indicators**: Indicators such as whether the stock price is above the 20-day Exponential Moving Average (EMA).
- **Target Variable**: The predicted market movement.

## Approach

### Data Preparation
- The dataset used in the project includes 393 data points, with monthly time intervals. Features such as retail sales change, sentiment, DJI index changes, and others are included.
- The dataset was imbalanced (with 70% positive market movements), which required techniques like **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the class distribution and avoid model bias.

### Model Selection and Training
- The model was trained for up to 50 epochs, using a batch size of 16.
- **Bi-directional GRU (BGRU)** was selected as the primary model for its ability to capture temporal dependencies in time-series data.
- **Regularization**: Techniques like **Dropout** and **L2 regularization** were implemented to avoid overfitting.
- **Early Stopping**: This was applied to stop training if the validation loss did not improve for 10 consecutive epochs.

### Evaluation
- The model was evaluated using **accuracy**, **precision**, **recall**, **F1-score**, and a **confusion matrix** to assess its classification performance.
- The model achieved a test accuracy of **71.62%**, with high recall for predicting the majority class (positive market movement).

## Results

- **Test Accuracy**: 71.62%
- **Best Model**: Bi-directional GRU (BGRU) achieved the highest accuracy at 72.97%.
- **Class Imbalance**: Despite using SMOTE and class weights to address the imbalance, the model still struggled with predicting the minority class (negative market movement).

## Key Findings
- The model performs well at predicting positive market movement but struggles to predict negative movements, likely due to class imbalance.
- Features like **Retail Sales Change** and **Sentiment** have a strong relationship with market movements, particularly when both are positive.
- Overfitting was a concern, particularly with deep learning models, due to the small dataset size.

## Future Directions
To improve model performance, the following approaches could be explored:
- **Additional Features**: Incorporating more macroeconomic indicators and technical indicators like RSI, MACD, and Bollinger Bands.
- **Real-Time Data**: Integrating real-time market data and news sentiment analysis to adapt to market changes.
- **Model Refinement**: Using advanced techniques like ensemble models or reinforcement learning to improve predictive accuracy.

## Installation

To use this repository, clone it to your local machine and install the required dependencies:

```bash
git clone git@github.com:r0han01/Market-Movement-Prediction-Using-Machine-Learning.git
cd market-prediction
