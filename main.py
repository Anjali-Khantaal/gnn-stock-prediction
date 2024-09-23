# main.py

from data_collection import collect_financial_data, build_company_graph, collect_news_sentiment
from data_preprocessing import preprocess_financial_data, preprocess_sentiment_data, normalize_features
from train import prepare_dataset, train_model
from evaluate import evaluate_model
from config import FEATURES

def main():
    try:
        # Data Collection
        print("Collecting financial data...")
        data = collect_financial_data()
        print("Financial data collected:", data)
        
        print("Building company graph...")
        G = build_company_graph()
        print("Company graph built:", G)
        
        print("Collecting news sentiment data...")
        news_sentiments = collect_news_sentiment()
        print("News sentiment data collected:", news_sentiments)
        
        # Data Preprocessing
        print("Preprocessing financial data...")
        data = preprocess_financial_data(data)
        print("Financial data preprocessed:", data)
        
        print("Preprocessing sentiment data...")
        data = preprocess_sentiment_data(data, news_sentiments)
        print("Sentiment data preprocessed:", data)
        
        print("Normalizing features...")
        data = normalize_features(data)
        print("Features normalized:", data)
        
        # Prepare Dataset
        print("Preparing dataset...")
        inputs, targets, edge_index = prepare_dataset(data, G, FEATURES)
        print("Dataset prepared:", inputs, targets, edge_index)
        
        # Train Model
        print("Training model...")
        input_dim = len(FEATURES)
        model = train_model(inputs, targets, edge_index, input_dim)
        print("Model trained:", model)
        
        # Evaluate Model
        print("Evaluating model...")
        evaluate_model(model, inputs, targets, edge_index)
        print("Model evaluated")
        
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    main()