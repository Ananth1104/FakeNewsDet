import json
import os
import requests
from bs4 import BeautifulSoup
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

def extract_article_content(url):
    """Extracts article text using BeautifulSoup with better error handling"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try different content extraction strategies
        paragraphs = soup.find_all("p")
        content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

        if not content:
            divs = soup.find_all("div")
            content = "\n".join([div.get_text(strip=True) for div in divs if div.get_text(strip=True)])

        if not content:
            print(f"Warning: No content extracted for {url}")
            return "Content not found"

        return content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return "Content not found"

def scrape_articles():
    """Reads news URLs and extracts full text"""
    file_path = os.path.join(DATA_DIR, "news_urls.json")
    
    if not os.path.exists(file_path):
        print("No news URLs found. Run news_fetcher.py first.")
        return False

    with open(file_path, "r", encoding="utf-8") as file:
        articles = json.load(file)

    full_articles = []
    failed_urls = []

    for article in articles:
        print(f"Scraping: {article['url']} ...")
        full_text = extract_article_content(article["url"])
        
        if full_text == "Content not found":
            failed_urls.append(article["url"])
        else:
            article["full_text"] = full_text
            full_articles.append(article)

        time.sleep(1)  # Avoid rate-limiting

    if not full_articles:
        print("All articles failed to scrape. Check URLs or network issues.")
        return False

    output_file = os.path.join(DATA_DIR, "raw_articles.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_articles, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully scraped {len(full_articles)} articles and saved to {output_file}")
    
    if failed_urls:
        print(f"\n {len(failed_urls)} articles failed to scrape. Check logs.")
    
    return True

if __name__ == "__main__":
    scrape_articles()
