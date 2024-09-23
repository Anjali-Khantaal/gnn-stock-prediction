# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import FEATURES

def preprocess_financial_data(data):
    for ticker in data:
        df = data[ticker]
        df['Returns'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df.dropna(inplace=True)
        data[ticker] = df
    return data

def preprocess_sentiment_data(data, news_sentiments):
    for ticker in data:
        sentiments = news_sentiments[ticker]
        sentiment_df = pd.DataFrame({'Date': sentiments['dates'], 'Sentiment': sentiments['sentiments']})
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
        sentiment_df = sentiment_df.groupby('Date').mean()
        df = data[ticker]
        df.index = pd.to_datetime(df.index)
        df = df.join(sentiment_df, how='left', on='Date')
        df['Sentiment'].fillna(0, inplace=True)
        data[ticker] = df
    return data

def normalize_features(data):
    scaler = StandardScaler()
    for ticker in data:
        df = data[ticker]
        df[FEATURES] = scaler.fit_transform(df[FEATURES])
        data[ticker] = df
    return data
