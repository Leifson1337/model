# src/sentiment_analysis.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch # Often needed by transformers, ensure it's available

from src import config # Assuming config.py is in the same directory or src is in PYTHONPATH

NEWS_CACHE_DIR = os.path.join(config.DATA_DIR, "news_cache")
os.makedirs(NEWS_CACHE_DIR, exist_ok=True)

def _get_cache_filename(ticker: str, start_date_str: str, end_date_str: str, page: int) -> str:
    """Generates a filename for caching NewsAPI responses."""
    return os.path.join(NEWS_CACHE_DIR, f"{ticker}_{start_date_str}_{end_date_str}_page{page}.json")

def _load_from_cache(filename: str):
    """Loads data from a cache file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading from cache file {filename}: {e}")
            return None
    return None

def _save_to_cache(data, filename: str):
    """Saves data to a cache file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving to cache file {filename}: {e}")

def fetch_news(
    ticker: str, 
    start_date_str: str, 
    end_date_str: str, 
    api_key: str = None, 
    max_articles_per_query: int = 100, # Max for 'everything' endpoint per request is 100
    max_pages: int = 1 # Limit pages to avoid excessive API calls
) -> list:
    """
    Fetches news articles for a given ticker and date range using NewsAPI.
    Implements file-based caching.
    """
    if api_key is None:
        api_key = config.NEWS_API_KEY
    
    if not api_key or api_key == "YOUR_NEWS_API_KEY_HERE":
        print("Warning: NewsAPI key not configured or is a placeholder. Skipping news fetching.")
        return []

    newsapi = NewsApiClient(api_key=api_key)
    all_articles_data = []
    
    print(f"Fetching news for {ticker} from {start_date_str} to {end_date_str}...")

    for page_num in range(1, max_pages + 1):
        cache_filename = _get_cache_filename(ticker, start_date_str, end_date_str, page_num)
        cached_data = _load_from_cache(cache_filename)

        if cached_data:
            print(f"Loading news from cache: {cache_filename}")
            articles_page = cached_data
        else:
            try:
                print(f"Fetching news from API - Page {page_num}...")
                articles_page = newsapi.get_everything(
                    q=ticker, # Query term
                    from_param=start_date_str,
                    to=end_date_str,
                    language='en',
                    sort_by='publishedAt', # 'relevancy' or 'popularity' or 'publishedAt'
                    page_size=max_articles_per_query,
                    page=page_num
                )
                if articles_page.get('status') == 'ok':
                    _save_to_cache(articles_page, cache_filename)
                else:
                    print(f"Error from NewsAPI: {articles_page.get('message')}")
                    return all_articles_data # Return what we have so far
            except Exception as e:
                print(f"Error fetching news from API: {e}")
                return all_articles_data # Return what we have so far
        
        if articles_page and articles_page.get('status') == 'ok':
            all_articles_data.extend(articles_page['articles'])
            if len(articles_page['articles']) < max_articles_per_query or page_num * max_articles_per_query >= articles_page.get('totalResults', 0):
                # Break if last page fetched or total results limit reached
                break 
        else: # If status not ok from cache or API
            break


    extracted_articles = []
    for article in all_articles_data:
        extracted_articles.append({
            'publishedAt': article.get('publishedAt'),
            'title': article.get('title'),
            'description': article.get('description'),
            'content': article.get('content') 
        })
    
    print(f"Fetched {len(extracted_articles)} total articles for {ticker}.")
    return extracted_articles


# Global sentiment pipeline (initialize once)
# Using a more general RoBERTa model fine-tuned for sentiment on Twitter, good for financial news too.
# Other options: "ProsusAI/finbert", "ElKulako/cryptobert-financial-sentiment" (more specific)
# "cardiffnlp/twitter-roberta-base-sentiment" outputs: 'positive', 'negative', 'neutral'
# If using FinBERT: labels are 'positive', 'negative', 'neutral'
# If using ElKulako: labels are 'Bearish', 'Bullish', 'Neutral'
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment" 
try:
    # Check if CUDA is available and set device accordingly
    device = 0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
    sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME, tokenizer=SENTIMENT_MODEL_NAME, device=device)
    print(f"Sentiment pipeline initialized with model: {SENTIMENT_MODEL_NAME} on device: {'cuda' if device==0 else 'cpu'}")
except Exception as e:
    print(f"Error initializing sentiment pipeline: {e}. Sentiment analysis will not work.")
    sentiment_pipeline = None


def score_sentiment_transformers(text_list: list) -> list:
    """
    Scores sentiment for a list of texts using a Hugging Face transformers pipeline.
    """
    if not sentiment_pipeline:
        print("Sentiment pipeline not available. Returning empty scores.")
        return [{'label': 'NEUTRAL', 'score': 0.0}] * len(text_list) # Default neutral if pipeline fails
        
    if not text_list:
        return []
    
    # Truncate texts to avoid issues with very long articles if model has input length limits
    # Max sequence length for RoBERTa is typically 512 tokens.
    # We can get tokenizer from pipeline or load it separately to check length, but simple truncation for now.
    # A better way is to use tokenizer's truncation=True, max_length=512 when calling pipeline.
    # However, pipeline itself might handle this. For safety, let's truncate input strings.
    MAX_TEXT_LENGTH = 500 # Characters, not tokens, as a rough estimate
    
    processed_texts = []
    for text in text_list:
        if not isinstance(text, str):
            processed_texts.append("") # Handle None or non-string inputs
        else:
            processed_texts.append(text[:MAX_TEXT_LENGTH])

    try:
        sentiments = sentiment_pipeline(processed_texts, truncation=True, max_length=512) # Added truncation
    except Exception as e:
        print(f"Error during sentiment pipeline processing: {e}")
        # Return neutral for all if pipeline fails mid-process
        sentiments = [{'label': 'NEUTRAL', 'score': 0.0}] * len(processed_texts)
        
    return sentiments


def get_daily_sentiment_scores(ticker: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Fetches news, scores sentiment, and aggregates daily sentiment scores.
    """
    articles = fetch_news(ticker, start_date_str, end_date_str)
    if not articles:
        return pd.DataFrame() # Return empty DataFrame if no articles

    texts_to_score = []
    publish_dates = []
    for article in articles:
        # Combine title and description for sentiment analysis
        title = article.get('title', '') or ""
        description = article.get('description', '') or ""
        text = title + ". " + description # Add a period for better sentence separation
        if not text.strip() or text == ". ": # If both are empty or just the period
            continue # Skip if no text content
            
        texts_to_score.append(text)
        publish_dates.append(article.get('publishedAt'))

    if not texts_to_score:
        print("No text content found in articles to score.")
        return pd.DataFrame()

    sentiments = score_sentiment_transformers(texts_to_score)

    # Map sentiment labels to numerical values
    # For "cardiffnlp/twitter-roberta-base-sentiment":
    # label 'positive' -> 1
    # label 'neutral'  -> 0
    # label 'negative' -> -1
    # For FinBERT: 'positive', 'negative', 'neutral' -> 1, -1, 0
    # For ElKulako: 'Bullish' -> 1, 'Bearish' -> -1, 'Neutral' -> 0
    
    sentiment_map = {
        'positive': 1, 'LABEL_2': 1, 'Bullish': 1, # Common positive labels
        'neutral': 0,  'LABEL_1': 0, 'Neutral': 0,  # Common neutral labels
        'negative': -1, 'LABEL_0': -1, 'Bearish': -1 # Common negative labels
    }
    
    numerical_sentiments = []
    sentiment_confidence_scores = []

    for sent in sentiments:
        # The label might be e.g. 'LABEL_2' or 'positive'. Normalize to lowercase for map.
        label_lower = sent['label'].lower()
        numerical_sentiments.append(sentiment_map.get(label_lower, 0)) # Default to neutral (0) if label not in map
        sentiment_confidence_scores.append(sent['score'])


    df = pd.DataFrame({
        'publishedAt': pd.to_datetime([d for d in publish_dates if d]), # Filter out None dates before conversion
        'numerical_sentiment': numerical_sentiments,
        'sentiment_score': sentiment_confidence_scores
    })
    
    if df.empty:
        return pd.DataFrame()

    df['date'] = df['publishedAt'].dt.normalize() # Normalize to day

    # Aggregate sentiment by day
    daily_sentiment = df.groupby('date').agg(
        mean_sentiment_polarity=('numerical_sentiment', 'mean'),
        median_sentiment_polarity=('numerical_sentiment', 'median'),
        std_sentiment_polarity=('numerical_sentiment', 'std'),
        min_sentiment_polarity=('numerical_sentiment', 'min'),
        max_sentiment_polarity=('numerical_sentiment', 'max'),
        mean_sentiment_score=('sentiment_score', 'mean'), # Avg confidence
        article_count=('numerical_sentiment', 'count')
    ).fillna(0) # Fill NaN for days with single article (std=NaN) with 0

    return daily_sentiment


if __name__ == '__main__':
    print("--- Sentiment Analysis Module Demonstration ---")
    # Ensure API key is set in config.py or as environment variable for this to run fully
    # Example: config.NEWS_API_KEY = "YOUR_ACTUAL_KEY" (do not commit real keys)
    
    if not config.NEWS_API_KEY or config.NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE":
        print("\nWARNING: NEWS_API_KEY is not set or is a placeholder in config.py.")
        print("News fetching and full sentiment analysis demo will be skipped.")
        print("Testing sentiment scoring with dummy text:")
        dummy_texts = [
            "Stocks are soaring high today!", 
            "Market is showing mixed signals.",
            "There are fears of an impending crash."
        ]
        dummy_sentiments = score_sentiment_transformers(dummy_texts)
        for text, sent in zip(dummy_texts, dummy_sentiments):
            print(f"Text: '{text}' -> Sentiment: {sent}")
    else:
        sample_ticker = config.DEFAULT_TICKERS[0] if config.DEFAULT_TICKERS else "AAPL"
        start_demo_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        end_demo_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"\n1. Fetching news for {sample_ticker} from {start_demo_date} to {end_demo_date} (max 1 page for demo)")
        # Limit max_pages for demo to avoid too many API calls
        articles_demo = fetch_news(sample_ticker, start_demo_date, end_demo_date, max_pages=1) 
        if articles_demo:
            print(f"Fetched {len(articles_demo)} articles. First few:")
            for i, art in enumerate(articles_demo[:2]):
                print(f"  Article {i+1}: {art['title']}")
        else:
            print("No articles fetched for demo.")

        print(f"\n2. Getting daily aggregated sentiment scores for {sample_ticker}...")
        daily_scores_df = get_daily_sentiment_scores(sample_ticker, start_demo_date, end_demo_date)
        if not daily_scores_df.empty:
            print("Daily Sentiment Scores (first 5 rows):")
            print(daily_scores_df.head())
        else:
            print("No daily sentiment scores generated.")
            
    print("\n--- End of Sentiment Analysis Module Demonstration ---")
