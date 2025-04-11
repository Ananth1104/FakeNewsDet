import os
import json
from scripts.news_fetcher import get_news_articles
from scripts.scraper import scrape_articles
from scripts.vectorizer import vectorize_articles
from scripts.similarity_checker_last_stable import check_similarity

def save_user_query(query):
    """Saves the user's query to data/user_input.json."""
    data_dir = os.path.join("data")
    os.makedirs(data_dir, exist_ok=True)
    user_input_path = os.path.join(data_dir, "user_input.json")
    data = {"user_query": query}
    with open(user_input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved query: {query} to {user_input_path}")

def main():
    # 1. Get user input and store it
    query = input("Enter search query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        return
    save_user_query(query)

    # 2. Fetch news articles from NewsAPI
    print("Fetching news articles...")
    articles = get_news_articles(query)
    if not articles:
        print("No articles fetched. Exiting.")
        return

    # 3. Scrape full text from fetched URLs
    print("Scraping articles...")
    if not scrape_articles():
        print("Scraping failed. Exiting.")
        return

    # 4. Vectorize the scraped articles and store embeddings in ChromaDB
    print("Vectorizing articles...")
    if not vectorize_articles():
        print("Vectorization failed. Exiting.")
        return

    # 5. Check similarity between the user query and stored articles.
    #    This will generate llm_input.json and then call the LLM summarizer.
    print("Checking similarity and generating LLM summary...")
    check_similarity()

if __name__ == "__main__":
    main()
