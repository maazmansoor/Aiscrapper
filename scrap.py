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

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except:
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
        self.use_crawl4ai = use_crawl4ai and CRAWL4AI_AVAILABLE
        
        if ai_config:
            if ai_config['type'] == 'gemini' and GEMINI_AVAILABLE and ai_config.get('key'):
                try:
                    genai.configure(api_key=ai_config['key'])
                    self.ai_client = genai.GenerativeModel('gemini-2.5-flash')
                    self.ai_type = 'gemini'
                except Exception as e:
                    st.warning(f"Gemini initialization failed: {e}")
                    
            elif ai_config['type'] == 'openai' and OPENAI_AVAILABLE and ai_config.get('key'):
                try:
                    self.ai_client = OpenAI(api_key=ai_config['key'])
                    self.ai_type = 'openai'
                except Exception as e:
                    st.warning(f"OpenAI initialization failed: {e}")
                    
            elif ai_config['type'] == 'anthropic' and ai_config.get('key'):
                self.anthropic_key = ai_config['key']
                self.ai_type = 'anthropic'
    
    def extract_with_crawl4ai(self, url):
        """Extract article content using Crawl4AI"""
        try:
            import asyncio
            from crawl4ai import AsyncWebCrawler
            
            async def crawl_url(target_url):
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=target_url)
                    return result
            
            result = asyncio.run(crawl_url(url))
            
            if result and result.markdown:
                text = result.markdown[:20000]
                if len(text) > 200:
                    return text
            
            if result and result.html:
                soup = BeautifulSoup(result.html, 'html.parser')
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()
                
                text = soup.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > 200:
                    return text[:20000]
        except Exception as e:
            st.warning(f"Crawl4AI error: {str(e)[:80]}")
        
        return None
    
    def get_actual_article_url(self, google_news_url):
        """Extract the real article URL from Google News redirect"""
        try:
            response = requests.get(google_news_url, headers=self.headers, timeout=10, allow_redirects=False)
            
            if response.status_code in [301, 302, 303, 307, 308]:
                actual_url = response.headers.get('Location', google_news_url)
                if actual_url and 'news.google.com' not in actual_url:
                    return actual_url
            
            response = requests.get(google_news_url, headers=self.headers, timeout=10, allow_redirects=True)
            actual_url = response.url
            
            if 'news.google.com' in actual_url:
                match = re.search(r'url=([^&]+)', actual_url)
                if match:
                    return unquote(match.group(1))
            
            if actual_url and 'news.google.com' not in actual_url:
                return actual_url
                
        except:
            pass
        
        return google_news_url
    
    def extract_article_content(self, url):
        """Extract article content from URL - PRIMARY METHOD"""
        
        if not url or 'news.google.com' in url:
            return None
        
        if TRAFILATURA_AVAILABLE:
            try:
                downloaded = trafilatura.fetch_url(url, timeout=15)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                    if text and len(text) > 200:
                        return text[:20000]
            except:
                pass
        
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url, request_timeout=15)
                article.download()
                article.parse()
                
                if article.text and len(article.text) > 200:
                    return article.text[:20000]
            except:
                pass
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript', 'meta', 'link', 'button', 'form']):
                element.decompose()
            
            article_content = None
            
            selectors = [
                'article', 'main', '[role="main"]', '.article-body', '.article-content',
                '.post-content', '.entry-content', '.story-body', '.news-body',
                '[itemprop="articleBody"]', '.article-text', '.content-main', '.prose',
                '[role="article"]', '.full-content', '.article__body', '.article-wrapper',
                '.story-content', '.post-body', '.blog-post', '.page-content', '.content',
                '.main-content', '[data-article]', '.article__content', '.news-article'
            ]
            
            for selector in selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        article_content = element
                        break
                except:
                    pass
            
            if not article_content:
                article_content = soup.find('body')
            
            if article_content:
                text_parts = []
                paragraphs = article_content.find_all('p', recursive=True)
                
                for para in paragraphs:
                    text = para.get_text(separator=' ', strip=True)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 30 and len(text) < 2000:
                        skip_keywords = ['cookie', 'advertisement', 'subscribe', 'javascript', 'loading', 'follow us', 'share this', 'read more', 'sign in', 'log in', 'menu', 'navigation', 'copyright', 'all rights']
                        if not any(skip in text.lower() for skip in skip_keywords):
                            if text not in text_parts:
                                text_parts.append(text)
                
                if len(text_parts) >= 3:
                    full_text = '\n\n'.join(text_parts[:150])
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    
                    if len(full_text) > 200:
                        return full_text[:20000]
                
                divs = article_content.find_all(['div', 'section'], recursive=True)
                for div in divs[:100]:
                    if div.find(['ul', 'ol', 'li', 'nav', 'header', 'footer', 'button']):
                        continue
                    
                    text = div.get_text(separator=' ', strip=True)
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    if len(text) > 30 and len(text) < 2000:
                        skip_keywords = ['cookie', 'advertisement', 'javascript', 'loading', 'menu']
                        if not any(skip in text.lower() for skip in skip_keywords):
                            if text not in text_parts:
                                text_parts.append(text)
                
                if text_parts:
                    full_text = '\n\n'.join(text_parts[:150])
                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                    
                    if len(full_text) > 200:
                        return full_text[:20000]
        
        except Exception as e:
            pass
        
        return None
    
    def extract_article_content_aggressive(self, url):
        """Aggressive extraction method for when normal methods fail"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript', 'meta', 'link', 'button', 'form']):
                element.decompose()
            
            text_parts = []
            all_paragraphs = soup.find_all(['p', 'div', 'section', 'article', 'span'])
            
            for para in all_paragraphs:
                text = para.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > 30 and len(text) < 2000:
                    skip_keywords = ['cookie', 'advertisement', 'javascript', 'loading', 'menu', 'navigation', 'subscribe', 'follow', 'share', 'copyright', 'all rights']
                    if not any(skip in text.lower() for skip in skip_keywords):
                        if 'http' not in text.lower() and '@' not in text:
                            if text not in text_parts:
                                text_parts.append(text)
            
            if text_parts:
                full_text = '\n\n'.join(text_parts[:200])
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                
                if len(full_text) > 200:
                    return full_text[:20000]
        
        except Exception as e:
            pass
        
        return None
    
    def summarize_with_ai(self, text):
        """Summarize content using AI - returns bullet points"""
        if not text or len(text) < 100 or not self.ai_client:
            return self.summarize_content(text)
        
        try:
            text_to_use = text[:5000]
            
            prompt = f"""Summarize this news article in 4-6 bullet points. Make each point concise and clear:

{text_to_use}

Bullet Points:"""
            
            if self.ai_type == 'gemini':
                try:
                    response = self.ai_client.generate_content(prompt)
                    result = response.text.strip()
                    if result:
                        return result
                except Exception as e:
                    st.warning(f"Gemini API error: {str(e)[:100]}")
                
            elif self.ai_type == 'openai':
                try:
                    response = self.ai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0.5
                    )
                    result = response.choices[0].message.content.strip()
                    if result:
                        return result
                except Exception as e:
                    st.warning(f"OpenAI API error: {str(e)[:100]}")
                
            elif self.ai_type == 'anthropic':
                try:
                    response = requests.post(
                        'https://api.anthropic.com/v1/messages',
                        headers={
                            'x-api-key': self.anthropic_key,
                            'anthropic-version': '2023-06-01',
                            'content-type': 'application/json'
                        },
                        json={
                            'model': 'claude-3-haiku-20240307',
                            'max_tokens': 300,
                            'messages': [{'role': 'user', 'content': prompt}]
                        },
                        timeout=15
                    )
                    if response.status_code == 200:
                        result = response.json()['content'][0]['text'].strip()
                        if result:
                            return result
                    else:
                        st.warning(f"Anthropic API error: {response.status_code}")
                except Exception as e:
                    st.warning(f"Anthropic API error: {str(e)[:100]}")
                
        except Exception as e:
            st.warning(f"AI Summarization error: {str(e)[:100]}")
        
        return self.summarize_content(text)
    
    def summarize_content(self, text, max_sentences=5):
        """Fallback simple summary - returns bullet points"""
        if not text:
            return "• No summary available"
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        bullet_points = '\n'.join([f"• {sent}" for sent in sentences[:max_sentences]])
        return bullet_points if bullet_points else "• No summary available"
    
    def clean_rss_description(self, description_tag):
        """Clean HTML from RSS description"""
        if not description_tag:
            return None
        
        try:
            raw = description_tag.decode_contents()
            raw = html.unescape(raw)
            clean = re.sub(r'<[^>]+>', '', raw)
            clean = re.sub(r'http\S+|www\S+', '', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            clean = re.sub(r'&nbsp;|&amp;|&quot;|&apos;|&lt;|&gt;', '', clean)
            clean = re.sub(r'\s+\.+\s*', '. ', clean)
            
            return clean if len(clean) > 20 else None
        except:
            return None
    
    def search_google_news(self, keyword, num_results=10, start_date=None, end_date=None, progress_callback=None):
        """Search Google News using RSS feed"""
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=en-US&gl=US&ceid=US:en"
            
            response = requests.get(rss_url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'xml')
            
            items = soup.find_all('item')[:num_results]
            
            if not items:
                st.warning(f"No results found for '{keyword}'")
                return
            
            for idx, item in enumerate(items):
                try:
                    title_tag = item.find('title')
                    link_tag = item.find('link')
                    pub_date_tag = item.find('pubDate')
                    description_tag = item.find('description')
                    
                    if not title_tag or not link_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    article_url = link_tag.get_text(strip=True)
                    pub_date = pub_date_tag.get_text(strip=True) if pub_date_tag else datetime.now().isoformat()
                    
                    try:
                        from email.utils import parsedate_to_datetime
                        article_datetime = parsedate_to_datetime(pub_date)
                        article_datetime = article_datetime.replace(tzinfo=None)
                    except:
                        article_datetime = datetime.now()
                    
                    if start_date and end_date:
                        if not (start_date <= article_datetime.date() <= end_date):
                            continue
                    
                    if progress_callback:
                        progress_callback(idx + 1, len(items), f"🔗 Fetching: {title[:50]}...")
                    
                    actual_url = self.get_actual_article_url(article_url)
                    
                    st.write(f"📥 Extracting from: {actual_url[:70]}...")
                    full_content = self.extract_article_content(actual_url)
                    
                    if not full_content or len(full_content) < 200:
                        full_content = self.extract_article_content_aggressive(actual_url)
                    
                    if self.use_crawl4ai and (not full_content or len(full_content) < 200):
                        st.write("🕷️ Using Crawl4AI for enhanced extraction...")
                        crawl_content = self.extract_with_crawl4ai(actual_url)
                        if crawl_content and len(crawl_content) > 200:
                            full_content = crawl_content
                    
                    if not full_content or len(full_content) < 150:
                        rss_desc = self.clean_rss_description(description_tag)
                        if rss_desc and len(rss_desc) > 150:
                            full_content = rss_desc
                    
                    if not full_content or len(full_content) < 150:
                        full_content = f"{title}\n\nUnable to extract full content. Source: {actual_url}"
                    
                    if len(full_content) > 150:
                        summary = self.summarize_with_ai(full_content)
                    else:
                        summary = self.summarize_content(full_content)
                    
                    article_data = {
                        'keyword': keyword,
                        'title': title,
                        'url': actual_url,
                        'date_time': article_datetime.isoformat(),
                        'full_content': full_content,
                        'summary': summary
                    }
                    
                    self.articles.append(article_data)
                    
                    if progress_callback:
                        progress_callback(idx + 1, len(items), f"✅ Done: {title[:50]}...")
                    
                    time.sleep(2)
                    
                except Exception as e:
                    st.warning(f"Error processing article: {str(e)[:80]}")
                    continue
                    
        except Exception as e:
            st.error(f"Error: {e}")
    
    def get_dataframe(self):
        """Convert to DataFrame"""
        if not self.articles:
            return None
        
        df = pd.DataFrame(self.articles)
        column_order = ['keyword', 'date_time', 'title', 'url', 'full_content', 'summary']
        df = df[column_order]
        df.columns = ['Keyword', 'Date/Time', 'Title', 'URL', 'Full Content', 'Summary']
        return df


def main():
    st.set_page_config(
        page_title="News Article Scraper",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Advanced News Article Scraper")
    st.markdown("*Extracts full article content & AI summaries from Google News*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("🕷️ Content Extraction")
        use_crawl4ai = st.checkbox("Enable Crawl4AI (Advanced extraction)", value=CRAWL4AI_AVAILABLE)
        if use_crawl4ai and not CRAWL4AI_AVAILABLE:
            st.warning("Crawl4AI not installed. Install with: `pip install crawl4ai`")
        
        st.subheader("🤖 AI Summaries (Optional)")
        use_ai_summary = st.checkbox("Enable AI Summaries", value=True)
        ai_config = None
        
        if use_ai_summary:
            ai_provider = st.selectbox(
                "AI Provider:",
                ["OpenAI (GPT-3.5)", "Gemini (Google)", "Claude (Anthropic)"]
            )
            
            ai_key = st.text_input(f"{ai_provider.split()[0]} API Key:", type="password")
            
            if ai_key:
                ai_map = {
                    "OpenAI (GPT-3.5)": "openai",
                    "Gemini (Google)": "gemini",
                    "Claude (Anthropic)": "anthropic"
                }
                ai_config = {'type': ai_map[ai_provider], 'key': ai_key}
        
        st.markdown("---")
        st.subheader("📅 Date Range")
        use_date_range = st.checkbox("Enable Date Filter", value=False)
        
        start_date, end_date = None, None
        if use_date_range:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("From:", value=datetime.now() - timedelta(days=7))
            with col2:
                end_date = st.date_input("To:", value=datetime.now())
            if start_date > end_date:
                st.error("Invalid date range!")
        
        st.markdown("---")
        st.subheader("🔍 Search Keywords")
        
        st.info("💡 **Tips:**\n- Simple: `Thailand`\n- Multiple words: `Bangkok news`\n- Use quotes for exact phrases")
        
        keywords_input = st.text_area(
            "Enter queries (one per line):",
            height=100,
            placeholder="Thailand news\nBangkok events\nAI Technology"
        )
        
        num_articles = st.slider("Articles per keyword:", 5, 30, 10, 5)
        search_button = st.button("🚀 Start Scraping", type="primary", use_container_width=True)
    
    if 'scraped_data' not in st.session_state:
        st.session_state.scraped_data = None
    if 'scraping_complete' not in st.session_state:
        st.session_state.scraping_complete = False
    
    if search_button:
        if not keywords_input.strip():
            st.error("Enter keywords!")
            return
        
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        scraper = NewsArticleScraper(ai_config=ai_config if use_ai_summary else None, use_crawl4ai=use_crawl4ai)
        
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        for idx, keyword in enumerate(keywords):
            status_text.markdown(f"### 🔍 Query: **{keyword}**")
            
            keyword_progress = st.progress(0)
            
            def progress_callback(current, total, title):
                keyword_progress.progress(min(current / total, 0.99))
            
            start_dt = datetime.combine(start_date, datetime.min.time()) if use_date_range and start_date else None
            end_dt = datetime.combine(end_date, datetime.max.time()) if use_date_range and end_date else None
            
            scraper.search_google_news(keyword, num_articles, start_dt, end_dt, progress_callback)
            
            overall_progress.progress((idx + 1) / len(keywords))
            keyword_progress.empty()
            time.sleep(1)
        
        status_text.success(f"✅ Done! {len(scraper.articles)} articles collected")
        overall_progress.empty()
        
        if scraper.articles:
            st.session_state.scraped_data = scraper.get_dataframe()
            st.session_state.scraping_complete = True
    
    if st.session_state.scraping_complete and st.session_state.scraped_data is not None:
        df = st.session_state.scraped_data
        
        st.markdown("---")
        st.header("📊 Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Articles", len(df))
        with col2:
            st.metric("Queries", df['Keyword'].nunique())
        with col3:
            st.metric("Avg", f"{len(df) / df['Keyword'].nunique():.1f}")
        with col4:
            st.metric("Type", "🤖 AI" if use_ai_summary else "📝 Basic")
        
        st.dataframe(df, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("💾 Download")
        
        col1, col2 = st.columns(2)
        with col1:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            st.download_button("📥 Excel", output.getvalue(), f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        with col2:
            st.download_button("📥 CSV", df.to_csv(index=False).encode(), f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        st.markdown("---")
        for idx, row in df.iterrows():
            with st.expander(f"📰 {row['Title'][:80]}..."):
                st.markdown(f"**Query:** {row['Keyword']} | **Date:** {str(row['Date/Time'])[:10]}")
                st.markdown(f"[🔗 Read]({row['URL']})")
                st.info(f"**Summary:** {row['Summary']}")
                if st.checkbox("Full content", key=f"_{idx}"):
                    st.text_area("Content", row['Full Content'], height=250, disabled=True, key=f"c_{idx}")
        
        if st.button("🗑️ Clear"):
            st.session_state.scraped_data = None
            st.session_state.scraping_complete = False
            st.rerun()
    
    elif not st.session_state.scraping_complete:
        st.info("👈 Configure and click 'Start Scraping'")

if __name__ == "__main__":
    main()
