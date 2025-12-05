import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
from urllib.parse import quote_plus, urlparse
import io
import re
import html

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# ------------------------------
# CLEAN URL EXTRACTOR (UPGRADED)
# ------------------------------
def clean_google_news_url(url):
    """Extracts the real article URL from Google News links, removing tracking parameters."""
    if not url:
        return ""

    try:
        if 'url=' in url:
            url = re.search(r"url=([^&]+)", url).group(1)
        elif 'articles/' in url:
            match = re.search(r"(https?://[^&]+)", url)
            if match:
                url = match.group(1)

        url = html.unescape(url)
        parsed = urlparse(url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return clean_url
    except:
        return url

# -----------------------------------------
# ARTICLE CONTENT SCRAPER (CRAWL4AI FIRST)
# -----------------------------------------
from crawl4ai import WebCrawler, CrawlerRunConfig

def extract_full_article(url):
    """Use crawl4ai first, then fallback to normal scraping."""
    # ---- Crawl4AI extraction ----
    try:
        crawler = WebCrawler()
        config = CrawlerRunConfig(bypass_cache=True)
        result = crawler.crawl(url=url, config=config)

        if result and hasattr(result, 'markdown') and result.markdown:
            text = result.markdown
            if len(text) > 200:
                return text
    except Exception:
        pass

    # ---- Fallback extraction ----
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        selectors = ["article", "main", "div[class*='content']", "section"]

        for sel in selectors:
            container = soup.select_one(sel)
            if container:
                text = container.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    return text

        all_p = " ".join([p.get_text(strip=True) for p in soup.find_all('p')])
        if len(all_p) > 200:
            return all_p

        return "Content could not be extracted."

    except Exception as e:
        return f"Error extracting content: {str(e)}"
# -----------------------------------------
def extract_full_article(url):
    """Heavily improved scraper: tries multiple selectors and fallback extraction."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        selectors = ["article", "main", "div[class*='content']", "section"]

        for sel in selectors:
            container = soup.select_one(sel)
            if container:
                text = container.get_text(separator=" ", strip=True)
                if len(text) > 200:
                    return text

        all_p = " ".join([p.get_text(strip=True) for p in soup.find_all('p')])
        if len(all_p) > 200:
            return all_p

        return "Content could not be extracted."

    except Exception as e:
        return f"Error extracting content: {str(e)}"

# ------------------------------
# AI SUMMARY (GEMINI OR GPT)
# ------------------------------
def generate_ai_summary(text, api_key, model, ai_choice):
    if not text or len(text) < 50:
        return "Not enough content to summarize."

    if ai_choice == "Gemini" and GEMINI_AVAILABLE:
        genai.configure(api_key=api_key)
        ai = genai.GenerativeModel(model)
        response = ai.generate_content(f"Summarize this news article in 5 bullet points:\n{text}")
        return response.text

    elif ai_choice == "OpenAI" and OPENAI_AVAILABLE:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Summarize this in bullets:\n{text}"}]
        )
        return response.choices[0].message.content

    return "AI model not available."

# ------------------------------
# GOOGLE NEWS SCRAPER (UPDATED)
# ------------------------------
def scrape_google_news(keyword, days):
    encoded_kw = quote_plus(keyword)
    url = f"https://news.google.com/search?q={encoded_kw}&hl=en-PK&gl=PK&ceid=PK%3Aen"

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    results = []

    for article in soup.select("article"):
        title_tag = article.select_one("h3 a")
        if not title_tag:
            continue

        title = title_tag.text.strip()
        link = "https://news.google.com" + title_tag.get("href", "")[1:]
        clean_url = clean_google_news_url(link)

        results.append({"title": title, "url": clean_url})

    return results

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("📰 Upgraded Google News Scraper + AI Summary")

keyword = st.text_input("Enter Keyword (Google News supports boolean operators):")
days = st.slider("Scrape last X days:", 1, 7, 1)
ai_choice = st.selectbox("Choose AI Model:", ["Gemini", "OpenAI"])        
api_key = st.text_input("Enter API Key:", type="password")
model = st.text_input("Model Name (e.g., gemini-2.0, gpt-4.1)")

if st.button("Scrape News"):
    st.write("Scraping Google News...")
    articles = scrape_google_news(keyword, days)

    final_output = []

    for art in articles:
        st.write(f"### {art['title']}")
        st.write(art['url'])

        content = extract_full_article(art['url'])
        summary = generate_ai_summary(content, api_key, model, ai_choice)

        final_output.append({
            "title": art['title'],
            "url": art['url'],
            "content": content,
            "summary": summary
        })

    df = pd.DataFrame(final_output)
    st.dataframe(df)
