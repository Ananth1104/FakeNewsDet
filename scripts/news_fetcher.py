import json
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_news_articles(query, num_articles=5):
    """Fetches articles using NewsAPI and saves them in data/raw_articles.json."""
    try:
        response = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=num_articles)

        if not response or "articles" not in response:
            print("❌ No articles found from API.")
            return []

        news_list = [
            {
                "title": article["title"],
                "description": article["description"],
                "content": article["content"],  
                "url": article["url"]
            }
            for article in response["articles"]
        ]

        # Save to data/raw_articles.json
        file_path = os.path.join(DATA_DIR, "news_urls.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(news_list, file, indent=4, ensure_ascii=False)

        print(f"Saved {len(news_list)} articles to {file_path}")
        return news_list

    except Exception as e:
        print(f"⚠️ Error fetching from API: {e}")
        return []

if __name__ == "__main__":
    query = input("Enter search query: ")
    get_news_articles(query)