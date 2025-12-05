import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
from urllib.parse import quote_plus, urlparse, unquote
import io
import re
import html

# Optional libs
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

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except:
    NEWSPAPER_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except:
    TRAFILATURA_AVAILABLE = False

# crawl4ai optional
try:
    from crawl4ai import WebCrawler
    CRAWL4AI_AVAILABLE = True
except Exception:
    CRAWL4AI_AVAILABLE = False


class NewsArticleScraper:
    def __init__(self, ai_config=None, use_crawl4ai=False):
        self.articles = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.google.com/'
        }
        self.ai_type = None
        self.ai_client = None
        # enable only if library available
        self.use_crawl4ai = use_crawl4ai and CRAWL4AI_AVAILABLE

    def extract_with_crawl4ai(self, url):
        """Use crawl4ai WebCrawler to fetch page and try to return {'url': canonical, 'content': text}"""
        if not CRAWL4AI_AVAILABLE:
            return None
        try:
            crawler = WebCrawler()
            result = crawler.crawl(url=url, bypass_cache=True)

            real_url = None
            content_text = None

            if not result:
                return None

            # Some crawl4ai results expose .url or .html or .markdown
            # Try markdown first
            if getattr(result, 'markdown', None):
                md = result.markdown
                if isinstance(md, str) and len(md) > 200:
                    content_text = md

            # Try html
            if not content_text and getattr(result, 'html', None):
                soup = BeautifulSoup(result.html, 'html.parser')

                # Find canonical or og:url
                can = soup.find('link', rel='canonical')
                if can and can.get('href'):
                    real_url = can.get('href')
                og = soup.find('meta', property='og:url')
                if not real_url and og and og.get('content'):
                    real_url = og.get('content')

                for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    el.decompose()

                paras = []
                for p in soup.find_all(['p', 'div']):
                    t = p.get_text(separator=' ', strip=True)
                    if t and len(t) > 40:
                        if not any(k in t.lower() for k in ['cookie', 'advertisement', 'subscribe', 'follow us', 'read more']):
                            paras.append(t)
                if paras:
                    content_text = '\n\n'.join(paras[:300])

            # If result has a top-level url attribute
            if getattr(result, 'url', None) and not real_url:
                try:
                    real_url = result.url
                except:
                    pass

            if content_text and len(content_text) > 200:
                return {'url': real_url or url, 'content': content_text[:20000]}

        except Exception as e:
            st.warning(f"crawl4ai error: {str(e)[:200]}")
        return None

    def get_actual_article_url(self, google_news_url):
        """Resolve google news redirect links to the real article url."""
        try:
            if not google_news_url:
                return google_news_url

            # If it's a google news rss 'link' it may include url= param
            if 'url=' in google_news_url:
                m = re.search(r'url=([^&]+)', google_news_url)
                if m:
                    return unquote(m.group(1))

            # handle relative paths from news.google.com
            if google_news_url.startswith('./') or google_news_url.startswith('/'):
                candidate = 'https://news.google.com' + google_news_url.lstrip('.')
            else:
                candidate = google_news_url

            # HEAD to follow redirects
            try:
                r = requests.head(candidate, headers=self.headers, allow_redirects=True, timeout=10)
                final = r.url
                if final and 'news.google.com' in final:
                    m = re.search(r'url=([^&]+)', final)
                    if m:
                        return unquote(m.group(1))
                return final or candidate
            except Exception:
                r = requests.get(candidate, headers=self.headers, allow_redirects=True, timeout=12)
                final = r.url
                if final and 'news.google.com' in final:
                    m = re.search(r'url=([^&]+)', final)
                    if m:
                        return unquote(m.group(1))
                return final or candidate
        except Exception:
            return google_news_url

    def extract_article_content(self, url):
        """Main extraction pipeline. If crawl4ai enabled it is used first (best for JS/paywall sites)."""
        if not url:
            return None

        # if still a google wrapper resolve it
        try:
            if 'news.google.com' in url or 'url=' in url:
                url = self.get_actual_article_url(url)
        except:
            pass

        # 1) Crawl4AI first (if enabled)
        if self.use_crawl4ai:
            try:
                crawl = self.extract_with_crawl4ai(url)
                if crawl and crawl.get('content') and len(crawl['content']) > 200:
                    return crawl['content']
                if crawl and crawl.get('url'):
                    url = crawl.get('url')
            except Exception:
                pass

        # 2) trafilatura
        if TRAFILATURA_AVAILABLE:
            try:
                downloaded = trafilatura.fetch_url(url, timeout=20)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                    if text and len(text) > 200:
                        return text[:20000]
            except Exception:
                pass

        # 3) newspaper
        if NEWSPAPER_AVAILABLE:
            try:
                art = Article(url, request_timeout=20)
                art.download()
                art.parse()
                if art.text and len(art.text) > 200:
                    return art.text[:20000]
            except Exception:
                pass

        # 4) manual BeautifulSoup extraction
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            resp.encoding = 'utf-8'
            soup = BeautifulSoup(resp.content, 'html.parser')
            for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                el.decompose()

            selectors = ['article', 'main', "[role='main']", '.article-body', '.entry-content', '.post-content', '.content']
            container = None
            for sel in selectors:
                try:
                    c = soup.select_one(sel)
                    if c:
                        container = c
                        break
                except:
                    continue

            if not container:
                container = soup.find('body')

            parts = []
            for p in container.find_all('p'):
                t = p.get_text(separator=' ', strip=True)
                t = re.sub(r'\s+', ' ', t).strip()
                if len(t) > 40 and not any(k in t.lower() for k in ['cookie', 'advertisement', 'subscribe']):
                    parts.append(t)
            if parts:
                full = '\n\n'.join(parts[:300])
                if len(full) > 200:
                    return full[:20000]

        except Exception:
            pass

        return None

    def extract_article_content_aggressive(self, url):
        try:
            resp = requests.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(resp.content, 'html.parser')
            for el in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                el.decompose()
            parts = []
            for node in soup.find_all(['p', 'div', 'section', 'article', 'span']):
                t = node.get_text(separator=' ', strip=True)
                t = re.sub(r'\s+', ' ', t).strip()
                if len(t) > 40 and 'http' not in t:
                    parts.append(t)
            if parts:
                return '\n\n'.join(parts[:400])[:20000]
        except Exception:
            pass
        return None

    def clean_rss_description(self, description_tag):
        if not description_tag:
            return None
        try:
            raw = description_tag.decode_contents()
            raw = html.unescape(raw)
            clean = re.sub(r'<[^>]+>', '', raw)
            clean = re.sub(r'http\S+|www\S+', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            return clean if len(clean) > 20 else None
        except:
            return None

    def summarize_with_ai(self, text):
        if not text:
            return '• No summary available'
        sents = re.split(r'[.!?]+', text)
        sents = [s.strip() for s in sents if len(s.strip()) > 20]
        return '\n'.join([f'• {s}' for s in sents[:5]]) if sents else '• No summary available'

    def search_google_news(self, keyword, num_results=10, start_date=None, end_date=None, progress_callback=None):
        rss_url = f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=en-US&gl=US&ceid=US:en"
        try:
            resp = requests.get(rss_url, headers=self.headers, timeout=12)
            soup = BeautifulSoup(resp.content, 'xml')
            items = soup.find_all('item')[:num_results]
            if not items:
                st.warning('No results')
                return

            for idx, item in enumerate(items):
                title = item.find('title').get_text(strip=True) if item.find('title') else 'No title'
                link = item.find('link').get_text(strip=True) if item.find('link') else None
                pub = item.find('pubDate').get_text(strip=True) if item.find('pubDate') else None
                desc = item.find('description')

                if not link:
                    continue

                actual = self.get_actual_article_url(link)
                if progress_callback:
                    progress_callback(idx+1, len(items), f'Fetching {title[:50]}')

                st.write(f'Attempting: {actual[:180]}')

                content = None
                # Try crawl4ai first if enabled
                if self.use_crawl4ai:
                    st.write('Using crawl4ai...')
                    try:
                        c = self.extract_with_crawl4ai(actual)
                        if c and c.get('content') and len(c.get('content')) > 200:
                            content = c.get('content')
                            actual = c.get('url') or actual
                    except Exception as e:
                        st.warning(f'crawl4ai extraction error: {str(e)[:120]}')

                if not content:
                    content = self.extract_article_content(actual)

                if not content or len(content) < 150:
                    content = self.extract_article_content_aggressive(actual)

                if (not content or len(content) < 150) and desc:
                    d = self.clean_rss_description(desc)
                    if d:
                        content = d

                if not content or len(content) < 150:
                    content = f"{title}\n\nUnable to extract full content. Source: {actual}"

                summary = self.summarize_with_ai(content)

                self.articles.append({
                    'keyword': keyword,
                    'title': title,
                    'url': actual,
                    'date_time': pub or datetime.now().isoformat(),
                    'full_content': content,
                    'summary': summary
                })

                time.sleep(1)

        except Exception as e:
            st.error(f'Error fetching RSS: {str(e)[:200]}')

    def get_dataframe(self):
        if not self.articles:
            return None
        df = pd.DataFrame(self.articles)
        df = df[['keyword', 'date_time', 'title', 'url', 'full_content', 'summary']]
        df.columns = ['Keyword', 'Date/Time', 'Title', 'URL', 'Full Content', 'Summary']
        return df


# ---------- Streamlit UI ----------

def main():
    st.set_page_config(page_title='News Scraper (crawl4ai)', layout='wide')
    st.title('News Scraper with crawl4ai Integration')

    with st.sidebar:
        use_crawl4ai = st.checkbox('Enable crawl4ai', value=CRAWL4AI_AVAILABLE)
        if use_crawl4ai and not CRAWL4AI_AVAILABLE:
            st.warning('crawl4ai not installed. pip install crawl4ai')
        keywords = st.text_area('Keywords (one per line)', height=120)
        num = st.slider('Articles per keyword', 5, 30, 10)
        start = st.button('Start')

    if start:
        if not keywords.strip():
            st.error('Enter keywords')
            return
        ks = [k.strip() for k in keywords.split('\n') if k.strip()]
        scraper = NewsArticleScraper(use_crawl4ai=use_crawl4ai)
        prog = st.progress(0)
        for i, k in enumerate(ks):
            scraper.search_google_news(k, num)
            prog.progress((i+1)/len(ks))
            time.sleep(0.5)

        df = scraper.get_dataframe()
        if df is not None:
            st.dataframe(df, height=400)
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='openpyxl') as w:
                df.to_excel(w, index=False)
            st.download_button('Download Excel', out.getvalue(), 'news.xlsx')


if __name__ == '__main__':
    main()
