# data_collection.py

import yfinance as yf
import networkx as nx
from transformers import pipeline
import requests
from config import TICKERS, START_DATE, END_DATE

def collect_financial_data():
    data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, interval='1d')
        data[ticker] = df
    return data

def build_company_graph():
    G = nx.Graph()
    G.add_nodes_from(TICKERS)
    # Simulate relationships
    edges = [
        ('AAPL', 'MSFT'),
        ('GOOGL', 'META'),
        # Add more edges as needed
    ]
    G.add_edges_from(edges)
    return G

def collect_news_sentiment():
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0)
    news_sentiments = {}
    api_key = '4402cacfb4c74d3ab6f407ae0d51c7cd'  # Replace with your NewsAPI key
    for ticker in TICKERS:
        sentiments, dates = get_news_sentiment(ticker, sentiment_pipeline, api_key)
        news_sentiments[ticker] = {'dates': dates, 'sentiments': sentiments}
    return news_sentiments

def get_news_sentiment(ticker, sentiment_pipeline, api_key):
    url = ('https://newsapi.org/v2/everything?'
           f'q={ticker}&'
           'sortBy=publishedAt&'
           'language=en&'
           f'apiKey={api_key}')
    response = requests.get(url)
    articles = response.json().get('articles', [])
    sentiments = []
    dates = []
    for article in articles:
        text = article['title']
        date = article['publishedAt'][:10]
        result = sentiment_pipeline(text)[0]
        score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        sentiments.append(score)
        dates.append(date)
    return sentiments, dates
