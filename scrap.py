# # # import streamlit as st
# # # import requests
# # # from bs4 import BeautifulSoup
# # # import pandas as pd
# # # from datetime import datetime, timedelta
# # # import time
# # # from urllib.parse import quote_plus
# # # import io
# # # try:
# # #     import google.generativeai as genai
# # #     GEMINI_AVAILABLE = True
# # # except:
# # #     GEMINI_AVAILABLE = False

# # # try:
# # #     from openai import OpenAI
# # #     OPENAI_AVAILABLE = True
# # # except:
# # #     OPENAI_AVAILABLE = False

# # # class NewsArticleScraper:
# # #     def __init__(self, ai_config=None):
# # #         self.articles = []
# # #         self.headers = {
# # #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# # #         }
# # #         self.ai_type = None
# # #         self.ai_client = None
        
# # #         if ai_config:
# # #             if ai_config['type'] == 'gemini' and GEMINI_AVAILABLE and ai_config.get('key'):
# # #                 try:
# # #                     genai.configure(api_key=ai_config['key'])
# # #                     # Try to find working model
# # #                     for model_name in ['gemini-pro', 'models/gemini-pro']:
# # #                         try:
# # #                             test_model = genai.GenerativeModel(model_name)
# # #                             test_model.generate_content("test")
# # #                             self.ai_client = test_model
# # #                             self.ai_type = 'gemini'
# # #                             break
# # #                         except:
# # #                             continue
# # #                 except Exception as e:
# # #                     st.warning(f"Gemini initialization failed: {e}")
                    
# # #             elif ai_config['type'] == 'openai' and OPENAI_AVAILABLE and ai_config.get('key'):
# # #                 try:
# # #                     self.ai_client = OpenAI(api_key=ai_config['key'])
# # #                     self.ai_type = 'openai'
# # #                 except Exception as e:
# # #                     st.warning(f"OpenAI initialization failed: {e}")
                    
# # #             elif ai_config['type'] == 'anthropic' and ai_config.get('key'):
# # #                 self.anthropic_key = ai_config['key']
# # #                 self.ai_type = 'anthropic'
    
# # #     def summarize_with_ai(self, text):
# # #         """Summarize content using AI (Gemini, OpenAI, or Anthropic)"""
# # #         if not text or len(text) < 50 or not self.ai_client:
# # #             return self.summarize_content(text)
        
# # #         try:
# # #             prompt = f"""Summarize the following news article in 3-4 concise sentences, highlighting the key points:

# # # {text[:4000]}"""
            
# # #             if self.ai_type == 'gemini':
# # #                 response = self.ai_client.generate_content(prompt)
# # #                 return response.text.strip()
                
# # #             elif self.ai_type == 'openai':
# # #                 response = self.ai_client.chat.completions.create(
# # #                     model="gpt-3.5-turbo",
# # #                     messages=[{"role": "user", "content": prompt}],
# # #                     max_tokens=200
# # #                 )
# # #                 return response.choices[0].message.content.strip()
                
# # #             elif self.ai_type == 'anthropic':
# # #                 response = requests.post(
# # #                     'https://api.anthropic.com/v1/messages',
# # #                     headers={
# # #                         'x-api-key': self.anthropic_key,
# # #                         'anthropic-version': '2023-06-01',
# # #                         'content-type': 'application/json'
# # #                     },
# # #                     json={
# # #                         'model': 'claude-3-haiku-20240307',
# # #                         'max_tokens': 200,
# # #                         'messages': [{'role': 'user', 'content': prompt}]
# # #                     }
# # #                 )
# # #                 return response.json()['content'][0]['text'].strip()
                
# # #         except Exception as e:
# # #             return self.summarize_content(text)
        
# # #         return self.summarize_content(text)
    
# # #     def search_google_news(self, keyword, num_results=10, start_date=None, end_date=None, progress_callback=None):
# # #         """Search Google News for articles based on keyword and date range"""
# # #         # Build search query with date filter
# # #         if start_date and end_date:
# # #             # Google News date format
# # #             search_query = f"{keyword} after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
# # #         else:
# # #             search_query = keyword
            
# # #         search_url = f"https://news.google.com/search?q={quote_plus(search_query)}&hl=en-US&gl=US&ceid=US:en"
        
# # #         try:
# # #             response = requests.get(search_url, headers=self.headers, timeout=10)
# # #             soup = BeautifulSoup(response.content, 'html.parser')
            
# # #             articles = soup.find_all('article')[:num_results]
            
# # #             for idx, article in enumerate(articles):
# # #                 try:
# # #                     link_tag = article.find('a', href=True)
# # #                     if link_tag:
# # #                         relative_url = link_tag['href']
# # #                         article_url = f"https://news.google.com{relative_url[1:]}"
# # #                         title = link_tag.get_text(strip=True)
                        
# # #                         time_tag = article.find('time')
# # #                         pub_date = time_tag.get('datetime') if time_tag else datetime.now().isoformat()
                        
# # #                         # Parse date and filter
# # #                         if start_date and end_date:
# # #                             try:
# # #                                 article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
# # #                                 if not (start_date <= article_date.replace(tzinfo=None) <= end_date + timedelta(days=1)):
# # #                                     continue
# # #                             except:
# # #                                 pass
                        
# # #                         content = self.extract_article_content(article_url)
                        
# # #                         # Use AI for summarization if available
# # #                         summary = self.summarize_with_ai(content if content else title)
                        
# # #                         article_data = {
# # #                             'keyword': keyword,
# # #                             'title': title,
# # #                             'url': article_url,
# # #                             'date_time': pub_date,
# # #                             'full_content': content if content else "Content extraction not available (redirect link)",
# # #                             'summary': summary
# # #                         }
                        
# # #                         self.articles.append(article_data)
                        
# # #                         if progress_callback:
# # #                             progress_callback(idx + 1, len(articles), title)
                        
# # #                         time.sleep(1)
                        
# # #                 except Exception as e:
# # #                     continue
                    
# # #         except Exception as e:
# # #             st.error(f"Error searching Google News: {e}")
    
# # #     def extract_article_content(self, url):
# # #         """Extract article content from URL"""
# # #         try:
# # #             response = requests.get(url, headers=self.headers, timeout=10)
# # #             soup = BeautifulSoup(response.content, 'html.parser')
            
# # #             content_selectors = ['article', '.article-content', '.post-content', 'main', '.content']
            
# # #             for selector in content_selectors:
# # #                 content = soup.select_one(selector)
# # #                 if content:
# # #                     paragraphs = content.find_all('p')
# # #                     text = ' '.join([p.get_text(strip=True) for p in paragraphs])
# # #                     if len(text) > 100:
# # #                         return text[:8000]
            
# # #             return None
            
# # #         except Exception as e:
# # #             return None
    
# # #     def summarize_content(self, text, max_sentences=3):
# # #         """Fallback: Create a simple summary by extracting first few sentences"""
# # #         if not text:
# # #             return "No summary available"
        
# # #         sentences = text.replace('!', '.').replace('?', '.').split('.')
# # #         sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
# # #         summary = '. '.join(sentences[:max_sentences])
# # #         return summary + '.' if summary else "No summary available"
    
# # #     def search_newsapi(self, keyword, api_key, num_results=10, start_date=None, end_date=None, progress_callback=None):
# # #         """Search using NewsAPI.org with date range"""
# # #         url = "https://newsapi.org/v2/everything"
# # #         params = {
# # #             'q': keyword,
# # #             'apiKey': api_key,
# # #             'language': 'en',
# # #             'sortBy': 'publishedAt',
# # #             'pageSize': num_results
# # #         }
        
# # #         # Add date range if provided
# # #         if start_date:
# # #             params['from'] = start_date.strftime('%Y-%m-%d')
# # #         if end_date:
# # #             params['to'] = end_date.strftime('%Y-%m-%d')
        
# # #         try:
# # #             response = requests.get(url, params=params, timeout=10)
# # #             data = response.json()
            
# # #             if data.get('status') == 'ok':
# # #                 articles_list = data.get('articles', [])
# # #                 for idx, article in enumerate(articles_list):
# # #                     content = article.get('content', article.get('description', 'No content available'))
                    
# # #                     # Use AI for summarization if available
# # #                     summary = self.summarize_with_ai(content)
                    
# # #                     article_data = {
# # #                         'keyword': keyword,
# # #                         'title': article.get('title', 'No title'),
# # #                         'url': article.get('url', ''),
# # #                         'date_time': article.get('publishedAt', ''),
# # #                         'full_content': content,
# # #                         'summary': summary
# # #                     }
# # #                     self.articles.append(article_data)
                    
# # #                     if progress_callback:
# # #                         progress_callback(idx + 1, len(articles_list), article_data['title'])
# # #             else:
# # #                 st.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
                
# # #         except Exception as e:
# # #             st.error(f"Error with NewsAPI: {e}")
    
# # #     def get_dataframe(self):
# # #         """Convert articles to DataFrame"""
# # #         if not self.articles:
# # #             return None
        
# # #         df = pd.DataFrame(self.articles)
# # #         column_order = ['keyword', 'date_time', 'title', 'url', 'full_content', 'summary']
# # #         df = df[column_order]
# # #         df.columns = ['Keyword', 'Date/Time', 'Title', 'URL', 'Full Content Extracted', 'Summarized Content']
# # #         return df


# # # # Streamlit App
# # # def main():
# # #     st.set_page_config(
# # #         page_title="News Article Scraper",
# # #         page_icon="📰",
# # #         layout="wide"
# # #     )
    
# # #     # Header
# # #     st.title("📰 Advanced News Article Scraper")
# # #     st.markdown("*with AI-Powered Summaries & Date Range Filtering*")
# # #     st.markdown("---")
    
# # #     # Sidebar configuration
# # #     with st.sidebar:
# # #         st.header("⚙️ Configuration")
        
# # #         # Method selection
# # #         method = st.radio(
# # #             "Select Scraping Method:",
# # #             ["Google News (Free)", "NewsAPI.org (API Key Required)"],
# # #             help="Google News is free but has limited content extraction. NewsAPI provides better content."
# # #         )
        
# # #         # API Keys Section
# # #         st.subheader("🔑 API Keys")
        
# # #         # NewsAPI key
# # #         news_api_key = None
# # #         if "NewsAPI" in method:
# # #             news_api_key = st.text_input(
# # #                 "NewsAPI Key:",
# # #                 type="password",
# # #                 help="Get your free API key from https://newsapi.org"
# # #             )
        
# # #         # AI Summarization Options
# # #         use_ai_summary = st.checkbox("🤖 Use AI for Better Summaries", value=False)
# # #         ai_config = None
        
# # #         if use_ai_summary:
# # #             ai_provider = st.selectbox(
# # #                 "Choose AI Provider:",
# # #                 ["OpenAI (GPT-3.5)", "Gemini (Google)", "Claude (Anthropic)"],
# # #                 help="OpenAI is most reliable, Gemini is free with limits, Claude is high quality"
# # #             )
            
# # #             ai_key = st.text_input(
# # #                 f"{ai_provider.split()[0]} API Key:",
# # #                 type="password",
# # #                 help={
# # #                     "OpenAI (GPT-3.5)": "Get from https://platform.openai.com/api-keys",
# # #                     "Gemini (Google)": "Get from https://ai.google.dev",
# # #                     "Claude (Anthropic)": "Get from https://console.anthropic.com"
# # #                 }[ai_provider]
# # #             )
            
# # #             if ai_key:
# # #                 ai_type_map = {
# # #                     "OpenAI (GPT-3.5)": "openai",
# # #                     "Gemini (Google)": "gemini",
# # #                     "Claude (Anthropic)": "anthropic"
# # #                 }
# # #                 ai_config = {'type': ai_type_map[ai_provider], 'key': ai_key}
            
# # #             st.caption("⭐ AI summaries are more accurate and contextual!")
        
# # #         st.markdown("---")
        
# # #         # Date Range Selection
# # #         st.subheader("📅 Date Range Filter")
# # #         use_date_range = st.checkbox("Enable Date Range Filter", value=False)
        
# # #         start_date = None
# # #         end_date = None
        
# # #         if use_date_range:
# # #             col1, col2 = st.columns(2)
# # #             with col1:
# # #                 start_date = st.date_input(
# # #                     "From Date:",
# # #                     value=datetime.now() - timedelta(days=7),
# # #                     max_value=datetime.now()
# # #                 )
# # #             with col2:
# # #                 end_date = st.date_input(
# # #                     "To Date:",
# # #                     value=datetime.now(),
# # #                     max_value=datetime.now()
# # #                 )
            
# # #             if start_date > end_date:
# # #                 st.error("Start date must be before end date!")
        
# # #         st.markdown("---")
        
# # #         # Keywords input
# # #         st.subheader("🔍 Search Parameters")
# # #         keywords_input = st.text_area(
# # #             "Enter Keywords (one per line):",
# # #             height=100,
# # #             placeholder="artificial intelligence\nmachine learning\nclimate change"
# # #         )
        
# # #         # Number of articles
# # #         num_articles = st.slider(
# # #             "Articles per keyword:",
# # #             min_value=5,
# # #             max_value=100,
# # #             value=20,
# # #             step=5
# # #         )
        
# # #         # Search button
# # #         search_button = st.button("🚀 Start Scraping", type="primary", use_container_width=True)
    
# # #     # Initialize session state for storing results
# # #     if 'scraped_data' not in st.session_state:
# # #         st.session_state.scraped_data = None
# # #     if 'scraping_complete' not in st.session_state:
# # #         st.session_state.scraping_complete = False
    
# # #     # Main content area
# # #     if search_button:
# # #         # Validate inputs
# # #         if not keywords_input.strip():
# # #             st.error("⚠️ Please enter at least one keyword!")
# # #             return
        
# # #         if "NewsAPI" in method and not news_api_key:
# # #             st.error("⚠️ Please enter your NewsAPI key!")
# # #             return
        
# # #         if use_date_range and start_date > end_date:
# # #             st.error("⚠️ Invalid date range!")
# # #             return
        
# # #         keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
# # #         # Initialize scraper with AI config
# # #         scraper = NewsArticleScraper(ai_config=ai_config if use_ai_summary else None)
        
# # #         # Show date range info
# # #         if use_date_range:
# # #             st.info(f"📅 Searching for articles from **{start_date.strftime('%d/%m/%Y')}** to **{end_date.strftime('%d/%m/%Y')}**")
        
# # #         # Progress tracking
# # #         overall_progress = st.progress(0)
# # #         status_text = st.empty()
        
# # #         # Scrape articles
# # #         total_keywords = len(keywords)
        
# # #         for idx, keyword in enumerate(keywords):
# # #             status_text.markdown(f"### 🔍 Searching for: **{keyword}**")
            
# # #             # Create progress bar for current keyword
# # #             keyword_progress = st.progress(0)
# # #             keyword_status = st.empty()
            
# # #             def progress_callback(current, total, title):
# # #                 progress = current / total
# # #                 keyword_progress.progress(progress)
# # #                 keyword_status.info(f"📄 Processing: {title[:60]}...")
            
# # #             # Convert dates to datetime if needed
# # #             start_dt = datetime.combine(start_date, datetime.min.time()) if use_date_range and start_date else None
# # #             end_dt = datetime.combine(end_date, datetime.max.time()) if use_date_range and end_date else None
            
# # #             # Scrape based on method
# # #             if "NewsAPI" in method:
# # #                 scraper.search_newsapi(keyword, news_api_key, num_articles, start_dt, end_dt, progress_callback)
# # #             else:
# # #                 scraper.search_google_news(keyword, num_articles, start_dt, end_dt, progress_callback)
            
# # #             # Update overall progress
# # #             overall_progress.progress((idx + 1) / total_keywords)
            
# # #             # Clear keyword progress
# # #             keyword_progress.empty()
# # #             keyword_status.empty()
            
# # #             time.sleep(1)
        
# # #         status_text.success(f"✅ Scraping complete! Collected {len(scraper.articles)} articles.")
# # #         overall_progress.empty()
        
# # #         # Store results in session state
# # #         if scraper.articles:
# # #             st.session_state.scraped_data = scraper.get_dataframe()
# # #             st.session_state.scraping_complete = True
# # #         else:
# # #             st.session_state.scraped_data = None
# # #             st.session_state.scraping_complete = False
    
# # #     # Display results (whether just scraped or from session state)
# # #     if st.session_state.scraping_complete and st.session_state.scraped_data is not None:
# # #         df = st.session_state.scraped_data
        
# # #         # Display results
# # #         st.markdown("---")
# # #         st.header("📊 Results")
            
# # #         # Display metrics
# # #         col1, col2, col3, col4 = st.columns(4)
# # #         with col1:
# # #             st.metric("Total Articles", len(df))
# # #         with col2:
# # #             keywords_count = df['Keyword'].nunique()
# # #             st.metric("Keywords Searched", keywords_count)
# # #         with col3:
# # #             st.metric("Avg per Keyword", f"{len(df) / keywords_count:.1f}")
# # #         with col4:
# # #             if use_ai_summary and ai_config:
# # #                 st.metric("Summary Type", "🤖 AI")
# # #             else:
# # #                 st.metric("Summary Type", "📝 Basic")
        
# # #         # Display dataframe
# # #         st.dataframe(df, use_container_width=True, height=400)
        
# # #         # Download options
# # #         st.markdown("---")
# # #         st.subheader("💾 Download Data")
        
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             # Excel download
# # #             output = io.BytesIO()
# # #             with pd.ExcelWriter(output, engine='openpyxl') as writer:
# # #                 df.to_excel(writer, index=False, sheet_name='News Articles')
# # #             excel_data = output.getvalue()
            
# # #             filename = f"news_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
# # #             st.download_button(
# # #                 label="📥 Download as Excel",
# # #                 data=excel_data,
# # #                 file_name=filename,
# # #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
# # #                 use_container_width=True
# # #             )
        
# # #         with col2:
# # #             # CSV download
# # #             csv_data = df.to_csv(index=False).encode('utf-8')
# # #             filename = f"news_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# # #             st.download_button(
# # #                 label="📥 Download as CSV",
# # #                 data=csv_data,
# # #                 file_name=filename,
# # #                 mime="text/csv",
# # #                 use_container_width=True
# # #             )
        
# # #         # Article previews
# # #         st.markdown("---")
# # #         st.subheader("📄 Article Previews")
        
# # #         for idx, row in df.iterrows():
# # #             with st.expander(f"📰 {row['Title'][:100]}..."):
# # #                 col1, col2 = st.columns([1, 3])
# # #                 with col1:
# # #                     st.markdown(f"**Keyword:** {row['Keyword']}")
# # #                     st.markdown(f"**Date:** {row['Date/Time'][:10]}")
# # #                 with col2:
# # #                     st.markdown(f"**URL:** [{row['URL']}]({row['URL']})")
                
# # #                 st.markdown("### 📝 AI Summary:")
# # #                 st.info(row['Summarized Content'])
                
# # #                 if st.checkbox(f"Show full content", key=f"content_{idx}"):
# # #                     st.text_area("Full Content:", row['Full Content Extracted'], height=300, key=f"full_{idx}")
        
# # #         # Clear results button
# # #         st.markdown("---")
# # #         if st.button("🗑️ Clear Results & Start New Search", type="secondary"):
# # #             st.session_state.scraped_data = None
# # #             st.session_state.scraping_complete = False
# # #             st.rerun()
    
# # #     elif search_button and st.session_state.scraped_data is None:
# # #         st.warning("No articles found for the specified criteria. Try different keywords, date range, or check your API key.")
    
# # #     # Show welcome screen only if no results
# # #     if not st.session_state.scraping_complete:
# # #         # Welcome screen
# # #         st.info("👈 Configure your search parameters in the sidebar and click 'Start Scraping' to begin!")
        
# # #         col1, col2 = st.columns(2)
        
# # #         with col1:
# # #             st.markdown("### 📖 How to Use:")
# # #             st.markdown("""
# # #             1. **Choose your scraping method**
# # #             2. **Enter API keys** (optional for AI summaries)
# # #             3. **Set date range** (optional)
# # #             4. **Enter keywords** (one per line)
# # #             5. **Click 'Start Scraping'**
# # #             6. **Download results** as Excel/CSV
# # #             """)
        
# # #         with col2:
# # #             st.markdown("### 🔑 Get Your API Keys:")
# # #             st.markdown("""
# # #             **NewsAPI** (Better content):
# # #             - Visit [newsapi.org](https://newsapi.org)
# # #             - 100 requests/day free
            
# # #             **AI Summaries** (Choose one):
# # #             - **OpenAI**: [platform.openai.com](https://platform.openai.com) (Most reliable)
# # #             - **Gemini**: [ai.google.dev](https://ai.google.dev) (Free with limits)
# # #             - **Claude**: [console.anthropic.com](https://console.anthropic.com) (High quality)
# # #             """)
        
# # #         st.markdown("---")
# # #         st.success("💡 **Tip:** Enable AI summaries for much better, contextual article summaries!")


# # # if __name__ == "__main__":
# # #     main()


# # import streamlit as st
# # import requests
# # from bs4 import BeautifulSoup
# # import pandas as pd
# # from datetime import datetime, timedelta
# # import time
# # from urllib.parse import quote_plus
# # import io
# # try:
# #     import google.generativeai as genai
# #     GEMINI_AVAILABLE = True
# # except:
# #     GEMINI_AVAILABLE = False

# # try:
# #     from openai import OpenAI
# #     OPENAI_AVAILABLE = True
# # except:
# #     OPENAI_AVAILABLE = False

# # try:
# #     from newspaper import Article
# #     NEWSPAPER_AVAILABLE = True
# # except:
# #     NEWSPAPER_AVAILABLE = False

# # try:
# #     import trafilatura
# #     TRAFILATURA_AVAILABLE = True
# # except:
# #     TRAFILATURA_AVAILABLE = False

# # class NewsArticleScraper:
# #     def __init__(self, ai_config=None):
# #         self.articles = []
# #         self.headers = {
# #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# #         }
# #         self.ai_type = None
# #         self.ai_client = None
        
# #         if ai_config:
# #             if ai_config['type'] == 'gemini' and GEMINI_AVAILABLE and ai_config.get('key'):
# #                 try:
# #                     genai.configure(api_key=ai_config['key'])
# #                     # Try to find working model
# #                     for model_name in ['gemini-pro', 'models/gemini-pro']:
# #                         try:
# #                             test_model = genai.GenerativeModel(model_name)
# #                             test_model.generate_content("test")
# #                             self.ai_client = test_model
# #                             self.ai_type = 'gemini'
# #                             break
# #                         except:
# #                             continue
# #                 except Exception as e:
# #                     st.warning(f"Gemini initialization failed: {e}")
                    
# #             elif ai_config['type'] == 'openai' and OPENAI_AVAILABLE and ai_config.get('key'):
# #                 try:
# #                     self.ai_client = OpenAI(api_key=ai_config['key'])
# #                     self.ai_type = 'openai'
# #                 except Exception as e:
# #                     st.warning(f"OpenAI initialization failed: {e}")
                    
# #             elif ai_config['type'] == 'anthropic' and ai_config.get('key'):
# #                 self.anthropic_key = ai_config['key']
# #                 self.ai_type = 'anthropic'
    
# #     def summarize_with_ai(self, text):
# #         """Summarize content using AI (Gemini, OpenAI, or Anthropic)"""
# #         if not text or len(text) < 50 or not self.ai_client:
# #             return self.summarize_content(text)
        
# #         try:
# #             # Use more of the text for better summaries
# #             text_to_summarize = text[:6000] if len(text) > 6000 else text
            
# #             prompt = f"""Summarize the following news article in 3-5 concise sentences, highlighting the key points and main facts:

# # {text_to_summarize}

# # Summary:"""
            
# #             if self.ai_type == 'gemini':
# #                 response = self.ai_client.generate_content(prompt)
# #                 return response.text.strip()
                
# #             elif self.ai_type == 'openai':
# #                 response = self.ai_client.chat.completions.create(
# #                     model="gpt-3.5-turbo",
# #                     messages=[{"role": "user", "content": prompt}],
# #                     max_tokens=300
# #                 )
# #                 return response.choices[0].message.content.strip()
                
# #             elif self.ai_type == 'anthropic':
# #                 response = requests.post(
# #                     'https://api.anthropic.com/v1/messages',
# #                     headers={
# #                         'x-api-key': self.anthropic_key,
# #                         'anthropic-version': '2023-06-01',
# #                         'content-type': 'application/json'
# #                     },
# #                     json={
# #                         'model': 'claude-3-haiku-20240307',
# #                         'max_tokens': 300,
# #                         'messages': [{'role': 'user', 'content': prompt}]
# #                     }
# #                 )
# #                 return response.json()['content'][0]['text'].strip()
                
# #         except Exception as e:
# #             return self.summarize_content(text)
        
# #         return self.summarize_content(text)
    
# #     def search_google_news(self, keyword, num_results=10, start_date=None, end_date=None, progress_callback=None):
# #         """Search Google News for articles based on keyword and date range"""
# #         # Build search query with date filter
# #         if start_date and end_date:
# #             # Google News date format
# #             search_query = f"{keyword} after:{start_date.strftime('%Y-%m-%d')} before:{end_date.strftime('%Y-%m-%d')}"
# #         else:
# #             search_query = keyword
            
# #         search_url = f"https://news.google.com/search?q={quote_plus(search_query)}&hl=en-US&gl=US&ceid=US:en"
        
# #         try:
# #             response = requests.get(search_url, headers=self.headers, timeout=10)
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             articles = soup.find_all('article')[:num_results]
            
# #             for idx, article in enumerate(articles):
# #                 try:
# #                     link_tag = article.find('a', href=True)
# #                     if link_tag:
# #                         relative_url = link_tag['href']
# #                         article_url = f"https://news.google.com{relative_url[1:]}"
# #                         title = link_tag.get_text(strip=True)
                        
# #                         time_tag = article.find('time')
# #                         pub_date = time_tag.get('datetime') if time_tag else datetime.now().isoformat()
                        
# #                         # Parse date and filter
# #                         if start_date and end_date:
# #                             try:
# #                                 article_date = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
# #                                 if not (start_date <= article_date.replace(tzinfo=None) <= end_date + timedelta(days=1)):
# #                                     continue
# #                             except:
# #                                 pass
                        
# #                         # Extract full content FIRST
# #                         if progress_callback:
# #                             progress_callback(idx + 1, len(articles), f"Extracting: {title[:40]}...")
                        
# #                         extracted_data = self.extract_article_content(article_url)
                        
# #                         # Get content and date
# #                         if extracted_data:
# #                             content = extracted_data['content']
# #                             extracted_date = extracted_data.get('date')
# #                             # Use extracted date if available and valid
# #                             if extracted_date:
# #                                 pub_date = extracted_date
# #                         else:
# #                             content = None
                        
# #                         # If extraction failed, try to get actual article URL from Google News redirect
# #                         if not content or len(content) < 200:
# #                             try:
# #                                 redirect_response = requests.get(article_url, headers=self.headers, timeout=5, allow_redirects=True)
# #                                 actual_url = redirect_response.url
# #                                 if actual_url != article_url and 'google' not in actual_url:
# #                                     extracted_data = self.extract_article_content(actual_url)
# #                                     if extracted_data:
# #                                         content = extracted_data['content']
# #                                         extracted_date = extracted_data.get('date')
# #                                         if extracted_date:
# #                                             pub_date = extracted_date
# #                                         article_url = actual_url
# #                             except:
# #                                 pass
                        
# #                         # Use full content for summary, or fallback to title
# #                         summary = self.summarize_with_ai(content if content and len(content) > 100 else title)
                        
# #                         article_data = {
# #                             'keyword': keyword,
# #                             'title': title,
# #                             'url': article_url,
# #                             'date_time': pub_date,
# #                             'full_content': content if content else f"[Content extraction unavailable]\n\nTitle: {title}",
# #                             'summary': summary
# #                         }
                        
# #                         self.articles.append(article_data)
                        
# #                         if progress_callback:
# #                             progress_callback(idx + 1, len(articles), title)
                        
# #                         time.sleep(2)  # Increased delay to be respectful
                        
# #                 except Exception as e:
# #                     continue
                    
# #         except Exception as e:
# #             st.error(f"Error searching Google News: {e}")
    
# #     def extract_article_content(self, url):
# #         """Extract article content using multiple methods for best results"""
        
# #         # Method 1: Try Newspaper3k (best for news articles)
# #         if NEWSPAPER_AVAILABLE:
# #             try:
# #                 article = Article(url)
# #                 article.download()
# #                 article.parse()
                
# #                 if article.text and len(article.text) > 200:
# #                     # Also get publish date if available
# #                     publish_date = article.publish_date
# #                     return {
# #                         'content': article.text[:15000],
# #                         'date': publish_date.isoformat() if publish_date else None
# #                     }
# #             except Exception as e:
# #                 pass
        
# #         # Method 2: Try Trafilatura (very good for article extraction)
# #         if TRAFILATURA_AVAILABLE:
# #             try:
# #                 downloaded = trafilatura.fetch_url(url)
# #                 if downloaded:
# #                     text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
# #                     if text and len(text) > 200:
# #                         # Try to extract date
# #                         metadata = trafilatura.extract_metadata(downloaded)
# #                         pub_date = metadata.date if metadata and hasattr(metadata, 'date') else None
# #                         return {
# #                             'content': text[:15000],
# #                             'date': pub_date
# #                         }
# #             except Exception as e:
# #                 pass
        
# #         # Method 3: BeautifulSoup with better extraction
# #         try:
# #             # Follow redirects for Google News links
# #             response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
# #             actual_url = response.url
# #             soup = BeautifulSoup(response.content, 'html.parser')
            
# #             # Try to extract date from meta tags
# #             pub_date = None
# #             date_selectors = [
# #                 {'property': 'article:published_time'},
# #                 {'name': 'pubdate'},
# #                 {'name': 'publishdate'},
# #                 {'name': 'date'},
# #                 {'itemprop': 'datePublished'}
# #             ]
            
# #             for selector in date_selectors:
# #                 date_tag = soup.find('meta', attrs=selector)
# #                 if date_tag and date_tag.get('content'):
# #                     pub_date = date_tag.get('content')
# #                     break
            
# #             # If no meta date, try time tags
# #             if not pub_date:
# #                 time_tag = soup.find('time')
# #                 if time_tag:
# #                     pub_date = time_tag.get('datetime') or time_tag.get_text(strip=True)
            
# #             # Remove unwanted elements
# #             for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
# #                 element.decompose()
            
# #             # Try multiple content extraction strategies
# #             content_text = None
            
# #             # Strategy 1: Article/main content tags
# #             content_selectors = [
# #                 'article',
# #                 '[role="main"]',
# #                 '.article-body',
# #                 '.article-content',
# #                 '.post-content',
# #                 '.entry-content',
# #                 '.content-body',
# #                 'main',
# #                 '.story-body',
# #                 '#article-body',
# #                 '.article__body',
# #                 '.post__content',
# #                 '[itemprop="articleBody"]'
# #             ]
            
# #             for selector in content_selectors:
# #                 content = soup.select_one(selector)
# #                 if content:
# #                     # Get all paragraphs
# #                     paragraphs = content.find_all(['p', 'div'], recursive=True)
# #                     text_parts = []
# #                     for p in paragraphs:
# #                         p_text = p.get_text(strip=True)
# #                         # Filter out short paragraphs (likely navigation/ads)
# #                         if len(p_text) > 40:
# #                             text_parts.append(p_text)
                    
# #                     text = '\n\n'.join(text_parts)
# #                     if len(text) > 300:
# #                         content_text = text
# #                         break
            
# #             # Strategy 2: Get all meaningful paragraphs
# #             if not content_text:
# #                 body = soup.find('body')
# #                 if body:
# #                     paragraphs = body.find_all('p')
# #                     text_parts = []
# #                     for p in paragraphs:
# #                         p_text = p.get_text(strip=True)
# #                         # Filter paragraphs
# #                         if len(p_text) > 40 and not any(skip in p_text.lower() for skip in ['cookie', 'subscribe', 'newsletter', 'advertisement']):
# #                             text_parts.append(p_text)
                    
# #                     text = '\n\n'.join(text_parts)
# #                     if len(text) > 300:
# #                         content_text = text
            
# #             if content_text and len(content_text) > 200:
# #                 return {
# #                     'content': content_text[:15000],
# #                     'date': pub_date
# #                 }
            
# #         except Exception as e:
# #             pass
        
# #         return None
    
# #     def summarize_content(self, text, max_sentences=3):
# #         """Fallback: Create a simple summary by extracting first few sentences"""
# #         if not text:
# #             return "No summary available"
        
# #         sentences = text.replace('!', '.').replace('?', '.').split('.')
# #         sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
# #         summary = '. '.join(sentences[:max_sentences])
# #         return summary + '.' if summary else "No summary available"
    
# #     def search_newsapi(self, keyword, api_key, num_results=10, start_date=None, end_date=None, progress_callback=None):
# #         """Search using NewsAPI.org with date range"""
# #         url = "https://newsapi.org/v2/everything"
# #         params = {
# #             'q': keyword,
# #             'apiKey': api_key,
# #             'language': 'en',
# #             'sortBy': 'publishedAt',
# #             'pageSize': num_results
# #         }
        
# #         # Add date range if provided
# #         if start_date:
# #             params['from'] = start_date.strftime('%Y-%m-%d')
# #         if end_date:
# #             params['to'] = end_date.strftime('%Y-%m-%d')
        
# #         try:
# #             response = requests.get(url, params=params, timeout=10)
# #             data = response.json()
            
# #             if data.get('status') == 'ok':
# #                 articles_list = data.get('articles', [])
# #                 for idx, article in enumerate(articles_list):
# #                     # Try to get full content from URL
# #                     article_url = article.get('url', '')
                    
# #                     if progress_callback:
# #                         progress_callback(idx + 1, len(articles_list), f"Extracting: {article.get('title', '')[:40]}...")
                    
# #                     extracted_data = self.extract_article_content(article_url)
                    
# #                     # Get content and date
# #                     full_content = None
# #                     extracted_date = None
                    
# #                     if extracted_data:
# #                         full_content = extracted_data['content']
# #                         extracted_date = extracted_data.get('date')
                    
# #                     # Fallback to NewsAPI content if extraction fails
# #                     if not full_content or len(full_content) < 200:
# #                         full_content = article.get('content', article.get('description', 'No content available'))
                    
# #                     # Use extracted date if available, otherwise use NewsAPI date
# #                     final_date = extracted_date if extracted_date else article.get('publishedAt', '')
                    
# #                     # Use full content for AI summarization
# #                     summary = self.summarize_with_ai(full_content)
                    
# #                     article_data = {
# #                         'keyword': keyword,
# #                         'title': article.get('title', 'No title'),
# #                         'url': article_url,
# #                         'date_time': final_date,
# #                         'full_content': full_content,
# #                         'summary': summary
# #                     }
# #                     self.articles.append(article_data)
                    
# #                     if progress_callback:
# #                         progress_callback(idx + 1, len(articles_list), article_data['title'])
                    
# #                     time.sleep(1)
# #             else:
# #                 st.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
                
# #         except Exception as e:
# #             st.error(f"Error with NewsAPI: {e}")
    
# #     def get_dataframe(self):
# #         """Convert articles to DataFrame"""
# #         if not self.articles:
# #             return None
        
# #         df = pd.DataFrame(self.articles)
# #         column_order = ['keyword', 'date_time', 'title', 'url', 'full_content', 'summary']
# #         df = df[column_order]
# #         df.columns = ['Keyword', 'Date/Time', 'Title', 'URL', 'Full Content Extracted', 'Summarized Content']
# #         return df


# # # Streamlit App
# # def main():
# #     st.set_page_config(
# #         page_title="News Article Scraper",
# #         page_icon="📰",
# #         layout="wide"
# #     )
    
# #     # Header
# #     st.title("📰 Advanced News Article Scraper")
# #     st.markdown("*with AI-Powered Summaries & Date Range Filtering*")
# #     st.markdown("---")
    
# #     # Sidebar configuration
# #     with st.sidebar:
# #         st.header("⚙️ Configuration")
        
# #         # Method selection
# #         method = st.radio(
# #             "Select Scraping Method:",
# #             ["Google News (Free)", "NewsAPI.org (API Key Required)"],
# #             help="Google News is free but has limited content extraction. NewsAPI provides better content."
# #         )
        
# #         # API Keys Section
# #         st.subheader("🔑 API Keys")
        
# #         # NewsAPI key
# #         news_api_key = None
# #         if "NewsAPI" in method:
# #             news_api_key = st.text_input(
# #                 "NewsAPI Key:",
# #                 type="password",
# #                 help="Get your free API key from https://newsapi.org"
# #             )
        
# #         # AI Summarization Options
# #         use_ai_summary = st.checkbox("🤖 Use AI for Better Summaries", value=False)
# #         ai_config = None
        
# #         if use_ai_summary:
# #             ai_provider = st.selectbox(
# #                 "Choose AI Provider:",
# #                 ["OpenAI (GPT-3.5)", "Gemini (Google)", "Claude (Anthropic)"],
# #                 help="OpenAI is most reliable, Gemini is free with limits, Claude is high quality"
# #             )
            
# #             ai_key = st.text_input(
# #                 f"{ai_provider.split()[0]} API Key:",
# #                 type="password",
# #                 help={
# #                     "OpenAI (GPT-3.5)": "Get from https://platform.openai.com/api-keys",
# #                     "Gemini (Google)": "Get from https://ai.google.dev",
# #                     "Claude (Anthropic)": "Get from https://console.anthropic.com"
# #                 }[ai_provider]
# #             )
            
# #             if ai_key:
# #                 ai_type_map = {
# #                     "OpenAI (GPT-3.5)": "openai",
# #                     "Gemini (Google)": "gemini",
# #                     "Claude (Anthropic)": "anthropic"
# #                 }
# #                 ai_config = {'type': ai_type_map[ai_provider], 'key': ai_key}
            
# #             st.caption("⭐ AI summaries are more accurate and contextual!")
        
# #         st.markdown("---")
        
# #         # Date Range Selection
# #         st.subheader("📅 Date Range Filter")
# #         use_date_range = st.checkbox("Enable Date Range Filter", value=False)
        
# #         start_date = None
# #         end_date = None
        
# #         if use_date_range:
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 start_date = st.date_input(
# #                     "From Date:",
# #                     value=datetime.now() - timedelta(days=7),
# #                     max_value=datetime.now()
# #                 )
# #             with col2:
# #                 end_date = st.date_input(
# #                     "To Date:",
# #                     value=datetime.now(),
# #                     max_value=datetime.now()
# #                 )
            
# #             if start_date > end_date:
# #                 st.error("Start date must be before end date!")
        
# #         st.markdown("---")
        
# #         # Keywords input
# #         st.subheader("🔍 Search Parameters")
# #         keywords_input = st.text_area(
# #             "Enter Keywords (one per line):",
# #             height=100,
# #             placeholder="artificial intelligence\nmachine learning\nclimate change"
# #         )
        
# #         # Number of articles
# #         num_articles = st.slider(
# #             "Articles per keyword:",
# #             min_value=5,
# #             max_value=100,
# #             value=20,
# #             step=5
# #         )
        
# #         # Search button
# #         search_button = st.button("🚀 Start Scraping", type="primary", use_container_width=True)
    
# #     # Initialize session state for storing results
# #     if 'scraped_data' not in st.session_state:
# #         st.session_state.scraped_data = None
# #     if 'scraping_complete' not in st.session_state:
# #         st.session_state.scraping_complete = False
    
# #     # Main content area
# #     if search_button:
# #         # Validate inputs
# #         if not keywords_input.strip():
# #             st.error("⚠️ Please enter at least one keyword!")
# #             return
        
# #         if "NewsAPI" in method and not news_api_key:
# #             st.error("⚠️ Please enter your NewsAPI key!")
# #             return
        
# #         if use_date_range and start_date > end_date:
# #             st.error("⚠️ Invalid date range!")
# #             return
        
# #         keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        
# #         # Initialize scraper with AI config
# #         scraper = NewsArticleScraper(ai_config=ai_config if use_ai_summary else None)
        
# #         # Show date range info
# #         if use_date_range:
# #             st.info(f"📅 Searching for articles from **{start_date.strftime('%d/%m/%Y')}** to **{end_date.strftime('%d/%m/%Y')}**")
        
# #         # Progress tracking
# #         overall_progress = st.progress(0)
# #         status_text = st.empty()
        
# #         # Scrape articles
# #         total_keywords = len(keywords)
        
# #         for idx, keyword in enumerate(keywords):
# #             status_text.markdown(f"### 🔍 Searching for: **{keyword}**")
            
# #             # Create progress bar for current keyword
# #             keyword_progress = st.progress(0)
# #             keyword_status = st.empty()
            
# #             def progress_callback(current, total, title):
# #                 progress = current / total
# #                 keyword_progress.progress(progress)
# #                 keyword_status.info(f"📄 Processing: {title[:60]}...")
            
# #             # Convert dates to datetime if needed
# #             start_dt = datetime.combine(start_date, datetime.min.time()) if use_date_range and start_date else None
# #             end_dt = datetime.combine(end_date, datetime.max.time()) if use_date_range and end_date else None
            
# #             # Scrape based on method
# #             if "NewsAPI" in method:
# #                 scraper.search_newsapi(keyword, news_api_key, num_articles, start_dt, end_dt, progress_callback)
# #             else:
# #                 scraper.search_google_news(keyword, num_articles, start_dt, end_dt, progress_callback)
            
# #             # Update overall progress
# #             overall_progress.progress((idx + 1) / total_keywords)
            
# #             # Clear keyword progress
# #             keyword_progress.empty()
# #             keyword_status.empty()
            
# #             time.sleep(1)
        
# #         status_text.success(f"✅ Scraping complete! Collected {len(scraper.articles)} articles.")
# #         overall_progress.empty()
        
# #         # Store results in session state
# #         if scraper.articles:
# #             st.session_state.scraped_data = scraper.get_dataframe()
# #             st.session_state.scraping_complete = True
# #         else:
# #             st.session_state.scraped_data = None
# #             st.session_state.scraping_complete = False
    
# #     # Display results (whether just scraped or from session state)
# #     if st.session_state.scraping_complete and st.session_state.scraped_data is not None:
# #         df = st.session_state.scraped_data
        
# #         # Display results
# #         st.markdown("---")
# #         st.header("📊 Results")
            
# #         # Display metrics
# #         col1, col2, col3, col4 = st.columns(4)
# #         with col1:
# #             st.metric("Total Articles", len(df))
# #         with col2:
# #             keywords_count = df['Keyword'].nunique()
# #             st.metric("Keywords Searched", keywords_count)
# #         with col3:
# #             st.metric("Avg per Keyword", f"{len(df) / keywords_count:.1f}")
# #         with col4:
# #             if use_ai_summary and ai_config:
# #                 st.metric("Summary Type", "🤖 AI")
# #             else:
# #                 st.metric("Summary Type", "📝 Basic")
        
# #         # Display dataframe
# #         st.dataframe(df, use_container_width=True, height=400)
        
# #         # Download options
# #         st.markdown("---")
# #         st.subheader("💾 Download Data")
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             # Excel download
# #             output = io.BytesIO()
# #             with pd.ExcelWriter(output, engine='openpyxl') as writer:
# #                 df.to_excel(writer, index=False, sheet_name='News Articles')
# #             excel_data = output.getvalue()
            
# #             filename = f"news_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
# #             st.download_button(
# #                 label="📥 Download as Excel",
# #                 data=excel_data,
# #                 file_name=filename,
# #                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
# #                 use_container_width=True
# #             )
        
# #         with col2:
# #             # CSV download
# #             csv_data = df.to_csv(index=False).encode('utf-8')
# #             filename = f"news_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# #             st.download_button(
# #                 label="📥 Download as CSV",
# #                 data=csv_data,
# #                 file_name=filename,
# #                 mime="text/csv",
# #                 use_container_width=True
# #             )
        
# #         # Article previews
# #         st.markdown("---")
# #         st.subheader("📄 Article Previews")
        
# #         for idx, row in df.iterrows():
# #             with st.expander(f"📰 {row['Title'][:100]}..."):
# #                 col1, col2 = st.columns([1, 3])
# #                 with col1:
# #                     st.markdown(f"**Keyword:** {row['Keyword']}")
# #                     st.markdown(f"**Date:** {row['Date/Time'][:10]}")
# #                 with col2:
# #                     st.markdown(f"**URL:** [{row['URL']}]({row['URL']})")
                
# #                 st.markdown("### 📝 AI Summary:")
# #                 st.info(row['Summarized Content'])
                
# #                 if st.checkbox(f"Show full content", key=f"content_{idx}"):
# #                     st.text_area("Full Content:", row['Full Content Extracted'], height=300, key=f"full_{idx}")
        
# #         # Clear results button
# #         st.markdown("---")
# #         if st.button("🗑️ Clear Results & Start New Search", type="secondary"):
# #             st.session_state.scraped_data = None
# #             st.session_state.scraping_complete = False
# #             st.rerun()
    
# #     elif search_button and st.session_state.scraped_data is None:
# #         st.warning("No articles found for the specified criteria. Try different keywords, date range, or check your API key.")
    
# #     # Show welcome screen only if no results
# #     if not st.session_state.scraping_complete:
# #         # Welcome screen
# #         st.info("👈 Configure your search parameters in the sidebar and click 'Start Scraping' to begin!")
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             st.markdown("### 📖 How to Use:")
# #             st.markdown("""
# #             1. **Choose your scraping method**
# #             2. **Enter API keys** (optional for AI summaries)
# #             3. **Set date range** (optional)
# #             4. **Enter keywords** (one per line)
# #             5. **Click 'Start Scraping'**
# #             6. **Download results** as Excel/CSV
            
# #             **📦 Recommended Libraries:**
# #             ```bash
# #             pip install newspaper3k
# #             pip install trafilatura
# #             ```
# #             These improve content extraction significantly!
# #             """)
        
# #         with col2:
# #             st.markdown("### 🔑 Get Your API Keys:")
# #             st.markdown("""
# #             **NewsAPI** (Better content):
# #             - Visit [newsapi.org](https://newsapi.org)
# #             - 100 requests/day free
            
# #             **AI Summaries** (Choose one):
# #             - **OpenAI**: [platform.openai.com](https://platform.openai.com) (Most reliable)
# #             - **Gemini**: [ai.google.dev](https://ai.google.dev) (Free with limits)
# #             - **Claude**: [console.anthropic.com](https://console.anthropic.com) (High quality)
# #             """)
        
# #         st.markdown("---")
# #         st.success("💡 **Tip:** Enable AI summaries for much better, contextual article summaries!")


# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
# from datetime import datetime, timedelta
# import time
# from urllib.parse import quote_plus, urlparse
# import io
# import re
# import html

# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
# except:
#     GEMINI_AVAILABLE = False

# try:
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except:
#     OPENAI_AVAILABLE = False

# try:
#     from newspaper import Article
#     NEWSPAPER_AVAILABLE = True
# except:
#     NEWSPAPER_AVAILABLE = False

# try:
#     import trafilatura
#     TRAFILATURA_AVAILABLE = True
# except:
#     TRAFILATURA_AVAILABLE = False

# class NewsArticleScraper:
#     def __init__(self, ai_config=None):
#         self.articles = []
#         self.headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
#             'Accept-Language': 'en-US,en;q=0.5',
#             'Accept-Encoding': 'gzip, deflate',
#             'Connection': 'keep-alive',
#             'Upgrade-Insecure-Requests': '1'
#         }
#         self.ai_type = None
#         self.ai_client = None
        
#         if ai_config:
#             if ai_config['type'] == 'gemini' and GEMINI_AVAILABLE and ai_config.get('key'):
#                 try:
#                     genai.configure(api_key=ai_config['key'])
#                     self.ai_client = genai.GenerativeModel('gemini-pro')
#                     self.ai_type = 'gemini'
#                 except Exception as e:
#                     st.warning(f"Gemini initialization failed: {e}")
                    
#             elif ai_config['type'] == 'openai' and OPENAI_AVAILABLE and ai_config.get('key'):
#                 try:
#                     self.ai_client = OpenAI(api_key=ai_config['key'])
#                     self.ai_type = 'openai'
#                 except Exception as e:
#                     st.warning(f"OpenAI initialization failed: {e}")
                    
#             elif ai_config['type'] == 'anthropic' and ai_config.get('key'):
#                 self.anthropic_key = ai_config['key']
#                 self.ai_type = 'anthropic'
    
#     def extract_article_content(self, url):
#         """Extract article content from URL - PRIMARY METHOD"""
        
#         if not url or url.startswith('http://news.google.com') or url.startswith('https://news.google.com'):
#             return None
        
#         # Method 1: Trafilatura (BEST for news)
#         if TRAFILATURA_AVAILABLE:
#             try:
#                 downloaded = trafilatura.fetch_url(url, timeout=15)
#                 if downloaded:
#                     text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
#                     if text and len(text) > 300:
#                         return text[:20000]
#             except:
#                 pass
        
#         # Method 2: Newspaper3k
#         if NEWSPAPER_AVAILABLE:
#             try:
#                 article = Article(url, request_timeout=15)
#                 article.download()
#                 article.parse()
                
#                 if article.text and len(article.text) > 300:
#                     return article.text[:20000]
#             except:
#                 pass
        
#         # Method 3: BeautifulSoup aggressive extraction
#         try:
#             response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
#             response.encoding = 'utf-8'
#             soup = BeautifulSoup(response.content, 'html.parser')
            
#             # Remove unwanted elements
#             for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript', 'meta', 'link', 'button']):
#                 element.decompose()
            
#             # Try to find article content
#             article_content = None
            
#             # Try specific article selectors first
#             selectors = [
#                 'article',
#                 'main',
#                 '[role="main"]',
#                 '.article-body',
#                 '.article-content',
#                 '.post-content',
#                 '.entry-content',
#                 '.story-body',
#                 '.news-body',
#                 '[itemprop="articleBody"]',
#                 '.article-text',
#                 '.content-main'
#             ]
            
#             for selector in selectors:
#                 element = soup.select_one(selector)
#                 if element:
#                     article_content = element
#                     break
            
#             # If no article element found, use body
#             if not article_content:
#                 article_content = soup.find('body')
            
#             if article_content:
#                 # Extract all paragraphs and meaningful text
#                 text_parts = []
                
#                 for para in article_content.find_all(['p', 'div', 'span']):
#                     text = para.get_text(strip=True)
                    
#                     # Filter: good length and not noise
#                     if len(text) > 40 and len(text) < 600:
#                         # Skip if it contains common noise
#                         skip_keywords = ['cookie', 'subscribe', 'advertisement', 'sign in', 'follow us', 'share this', 'read more', 'javascript', 'loading', 'more videos']
#                         if not any(skip in text.lower() for skip in skip_keywords):
#                             text_parts.append(text)
                
#                 if text_parts:
#                     full_text = '\n\n'.join(text_parts[:50])
#                     if len(full_text) > 300:
#                         return full_text[:20000]
        
#         except:
#             pass
        
#         return None
    
#     def summarize_with_ai(self, text):
#         """Summarize content using AI"""
#         if not text or len(text) < 100 or not self.ai_client:
#             return self.summarize_content(text)
        
#         try:
#             text_to_use = text[:5000]
            
#             prompt = f"""Summarize this news article in 3-5 clear sentences:

# {text_to_use}

# Summary:"""
            
#             if self.ai_type == 'gemini':
#                 response = self.ai_client.generate_content(prompt, timeout=10)
#                 return response.text.strip()
                
#             elif self.ai_type == 'openai':
#                 response = self.ai_client.chat.completions.create(
#                     model="gpt-3.5-turbo",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=250,
#                     timeout=10
#                 )
#                 return response.choices[0].message.content.strip()
                
#             elif self.ai_type == 'anthropic':
#                 response = requests.post(
#                     'https://api.anthropic.com/v1/messages',
#                     headers={
#                         'x-api-key': self.anthropic_key,
#                         'anthropic-version': '2023-06-01',
#                         'content-type': 'application/json'
#                     },
#                     json={
#                         'model': 'claude-3-haiku-20240307',
#                         'max_tokens': 250,
#                         'messages': [{'role': 'user', 'content': prompt}]
#                     },
#                     timeout=10
#                 )
#                 return response.json()['content'][0]['text'].strip()
                
#         except Exception as e:
#             pass
        
#         return self.summarize_content(text)
    
#     def summarize_content(self, text, max_sentences=3):
#         """Fallback simple summary"""
#         if not text:
#             return "No summary available"
        
#         sentences = re.split(r'[.!?]+', text)
#         sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
#         summary = '. '.join(sentences[:max_sentences])
#         return (summary + '.') if summary else "No summary available"
    
#     def clean_rss_description(self, description_tag):
#         """Clean HTML from RSS description"""
#         if not description_tag:
#             return None
        
#         try:
#             raw = description_tag.decode_contents()
#             raw = html.unescape(raw)
#             clean = re.sub(r'<[^>]+>', '', raw)
#             clean = re.sub(r'http\S+|www\S+', '', clean)
#             clean = re.sub(r'\s+', ' ', clean).strip()
#             clean = re.sub(r'&nbsp;|&amp;|&quot;|&apos;', '', clean)
            
#             return clean if len(clean) > 20 else None
#         except:
#             return None
    
#     def search_google_news(self, keyword, num_results=10, start_date=None, end_date=None, progress_callback=None):
#         """Search Google News using RSS feed"""
#         try:
#             rss_url = f"https://news.google.com/rss/search?q={quote_plus(keyword)}&hl=en-US&gl=US&ceid=US:en"
            
#             response = requests.get(rss_url, headers=self.headers, timeout=15)
#             soup = BeautifulSoup(response.content, 'xml')
            
#             items = soup.find_all('item')[:num_results]
            
#             if not items:
#                 st.warning(f"No results found for '{keyword}'")
#                 return
            
#             for idx, item in enumerate(items):
#                 try:
#                     title_tag = item.find('title')
#                     link_tag = item.find('link')
#                     pub_date_tag = item.find('pubDate')
#                     description_tag = item.find('description')
                    
#                     if not title_tag or not link_tag:
#                         continue
                    
#                     title = title_tag.get_text(strip=True)
#                     article_url = link_tag.get_text(strip=True)
#                     pub_date = pub_date_tag.get_text(strip=True) if pub_date_tag else datetime.now().isoformat()
                    
#                     # Parse date
#                     try:
#                         from email.utils import parsedate_to_datetime
#                         article_datetime = parsedate_to_datetime(pub_date)
#                         article_datetime = article_datetime.replace(tzinfo=None)
#                     except:
#                         article_datetime = datetime.now()
                    
#                     # Filter by date
#                     if start_date and end_date:
#                         if not (start_date <= article_datetime.date() <= end_date):
#                             continue
                    
#                     # UPDATE PROGRESS
#                     if progress_callback:
#                         progress_callback(idx + 1, len(items), f"🔗 Fetching: {title[:50]}...")
                    
#                     # EXTRACT FULL ARTICLE CONTENT FROM URL
#                     st.write(f"📥 Extracting content from: {article_url[:60]}...")
#                     full_content = self.extract_article_content(article_url)
                    
#                     # FALLBACK to RSS description if extraction fails
#                     if not full_content or len(full_content) < 200:
#                         full_content = self.clean_rss_description(description_tag)
                    
#                     if not full_content:
#                         full_content = title
                    
#                     # GENERATE SUMMARY from extracted content
#                     summary = self.summarize_with_ai(full_content) if len(full_content) > 100 else self.summarize_content(full_content)
                    
#                     article_data = {
#                         'keyword': keyword,
#                         'title': title,
#                         'url': article_url,
#                         'date_time': article_datetime.isoformat(),
#                         'full_content': full_content,
#                         'summary': summary
#                     }
                    
#                     self.articles.append(article_data)
                    
#                     if progress_callback:
#                         progress_callback(idx + 1, len(items), f"✅ Done: {title[:50]}...")
                    
#                     time.sleep(2)
                    
#                 except Exception as e:
#                     continue
                    
#         except Exception as e:
#             st.error(f"Error: {e}")
    
#     def search_newsapi(self, keyword, api_key, num_results=10, start_date=None, end_date=None, progress_callback=None):
#         """Search using NewsAPI.org"""
#         url = "https://newsapi.org/v2/everything"
#         params = {
#             'q': keyword,
#             'apiKey': api_key,
#             'language': 'en',
#             'sortBy': 'publishedAt',
#             'pageSize': num_results
#         }
        
#         if start_date:
#             params['from'] = start_date.strftime('%Y-%m-%d')
#         if end_date:
#             params['to'] = end_date.strftime('%Y-%m-%d')
        
#         try:
#             response = requests.get(url, params=params, timeout=10)
#             data = response.json()
            
#             if data.get('status') == 'ok':
#                 articles_list = data.get('articles', [])
#                 for idx, article in enumerate(articles_list):
#                     article_url = article.get('url', '')
                    
#                     if progress_callback:
#                         progress_callback(idx + 1, len(articles_list), f"🔗 Fetching: {article.get('title', '')[:50]}...")
                    
#                     # Extract full content from URL
#                     st.write(f"📥 Extracting: {article_url[:60]}...")
#                     full_content = self.extract_article_content(article_url)
                    
#                     # Fallback to NewsAPI content
#                     if not full_content or len(full_content) < 200:
#                         full_content = article.get('content', article.get('description', article.get('title', 'No content')))
                    
#                     # Generate summary
#                     summary = self.summarize_with_ai(full_content)
                    
#                     article_data = {
#                         'keyword': keyword,
#                         'title': article.get('title', 'No title'),
#                         'url': article_url,
#                         'date_time': article.get('publishedAt', ''),
#                         'full_content': full_content,
#                         'summary': summary
#                     }
#                     self.articles.append(article_data)
                    
#                     if progress_callback:
#                         progress_callback(idx + 1, len(articles_list), article_data['title'][:50])
                    
#                     time.sleep(1)
#             else:
#                 st.error(f"NewsAPI Error: {data.get('message', 'Unknown')}")
                
#         except Exception as e:
#             st.error(f"Error: {e}")
    
#     def get_dataframe(self):
#         """Convert to DataFrame"""
#         if not self.articles:
#             return None
        
#         df = pd.DataFrame(self.articles)
#         column_order = ['keyword', 'date_time', 'title', 'url', 'full_content', 'summary']
#         df = df[column_order]
#         df.columns = ['Keyword', 'Date/Time', 'Title', 'URL', 'Full Content', 'Summary']
#         return df


# def main():
#     st.set_page_config(
#         page_title="News Article Scraper",
#         page_icon="📰",
#         layout="wide"
#     )
    
#     st.title("📰 Advanced News Article Scraper")
#     st.markdown("*Extracts full article content & AI summaries*")
#     st.markdown("---")
    
#     with st.sidebar:
#         st.header("⚙️ Configuration")
        
#         method = st.radio(
#             "Select Method:",
#             ["Google News (Free - RSS)", "NewsAPI.org (API Key)"],
#             help="Both will extract full article content from source URLs"
#         )
        
#         st.subheader("🔑 API Keys")
        
#         news_api_key = None
#         if "NewsAPI" in method:
#             news_api_key = st.text_input("NewsAPI Key:", type="password")
        
#         use_ai_summary = st.checkbox("🤖 Enable AI Summaries", value=True)
#         ai_config = None
        
#         if use_ai_summary:
#             ai_provider = st.selectbox(
#                 "AI Provider:",
#                 ["OpenAI (GPT-3.5)", "Gemini (Google)", "Claude (Anthropic)"]
#             )
            
#             ai_key = st.text_input(f"{ai_provider.split()[0]} Key:", type="password")
            
#             if ai_key:
#                 ai_map = {
#                     "OpenAI (GPT-3.5)": "openai",
#                     "Gemini (Google)": "gemini",
#                     "Claude (Anthropic)": "anthropic"
#                 }
#                 ai_config = {'type': ai_map[ai_provider], 'key': ai_key}
        
#         st.markdown("---")
#         st.subheader("📅 Date Range")
#         use_date_range = st.checkbox("Enable Date Filter", value=False)
        
#         start_date, end_date = None, None
#         if use_date_range:
#             col1, col2 = st.columns(2)
#             with col1:
#                 start_date = st.date_input("From:", value=datetime.now() - timedelta(days=7))
#             with col2:
#                 end_date = st.date_input("To:", value=datetime.now())
#             if start_date > end_date:
#                 st.error("Invalid date range!")
        
#         st.markdown("---")
#         st.subheader("🔍 Search")
#         keywords_input = st.text_area(
#             "Keywords (one per line):",
#             height=100,
#             placeholder="AI\nTechnology\nNews"
#         )
        
#         num_articles = st.slider("Articles per keyword:", 5, 30, 10, 5)
#         search_button = st.button("🚀 Start Scraping", type="primary", use_container_width=True)
    
#     if 'scraped_data' not in st.session_state:
#         st.session_state.scraped_data = None
#     if 'scraping_complete' not in st.session_state:
#         st.session_state.scraping_complete = False
    
#     if search_button:
#         if not keywords_input.strip():
#             st.error("Enter keywords!")
#             return
        
#         if "NewsAPI" in method and not news_api_key:
#             st.error("Enter NewsAPI key!")
#             return
        
#         keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
#         scraper = NewsArticleScraper(ai_config=ai_config if use_ai_summary else None)
        
#         overall_progress = st.progress(0)
#         status_text = st.empty()
        
#         for idx, keyword in enumerate(keywords):
#             status_text.markdown(f"### 🔍 Keyword: **{keyword}**")
            
#             keyword_progress = st.progress(0)
            
#             def progress_callback(current, total, title):
#                 keyword_progress.progress(min(current / total, 0.99))
            
#             start_dt = datetime.combine(start_date, datetime.min.time()) if use_date_range and start_date else None
#             end_dt = datetime.combine(end_date, datetime.max.time()) if use_date_range and end_date else None
            
#             if "NewsAPI" in method:
#                 scraper.search_newsapi(keyword, news_api_key, num_articles, start_dt, end_dt, progress_callback)
#             else:
#                 scraper.search_google_news(keyword, num_articles, start_dt, end_dt, progress_callback)
            
#             overall_progress.progress((idx + 1) / len(keywords))
#             keyword_progress.empty()
#             time.sleep(1)
        
#         status_text.success(f"✅ Done! {len(scraper.articles)} articles collected")
#         overall_progress.empty()
        
#         if scraper.articles:
#             st.session_state.scraped_data = scraper.get_dataframe()
#             st.session_state.scraping_complete = True
    
#     if st.session_state.scraping_complete and st.session_state.scraped_data is not None:
#         df = st.session_state.scraped_data
        
#         st.markdown("---")
#         st.header("📊 Results")
        
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Articles", len(df))
#         with col2:
#             st.metric("Keywords", df['Keyword'].nunique())
#         with col3:
#             st.metric("Avg", f"{len(df) / df['Keyword'].nunique():.1f}")
#         with col4:
#             st.metric("Type", "🤖 AI" if use_ai_summary else "📝 Basic")
        
#         st.dataframe(df, use_container_width=True, height=400)
        
#         st.markdown("---")
#         st.subheader("💾 Download")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             output = io.BytesIO()
#             with pd.ExcelWriter(output, engine='openpyxl') as writer:
#                 df.to_excel(writer, index=False)
#             st.download_button("📥 Excel", output.getvalue(), f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
#         with col2:
#             st.download_button("📥 CSV", df.to_csv(index=False).encode(), f"news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
#         st.markdown("---")
#         for idx, row in df.iterrows():
#             with st.expander(f"📰 {row['Title'][:80]}..."):
#                 st.markdown(f"**Keyword:** {row['Keyword']} | **Date:** {str(row['Date/Time'])[:10]}")
#                 st.markdown(f"[🔗 Read]({row['URL']})")
#                 st.info(f"**Summary:** {row['Summary']}")
#                 if st.checkbox("Full content", key=f"_{idx}"):
#                     st.text_area("Content", row['Full Content'], height=250, disabled=True, key=f"c_{idx}")
        
#         if st.button("🗑️ Clear"):
#             st.session_state.scraped_data = None
#             st.session_state.scraping_complete = False
#             st.rerun()
    
#     elif not st.session_state.scraping_complete:
#         st.info("👈 Configure and click Scrape")


# if __name__ == "__main__":
#     main()

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

class NewsArticleScraper:
    def __init__(self, ai_config=None):
        self.articles = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.ai_type = None
        self.ai_client = None
        
        if ai_config:
            if ai_config['type'] == 'gemini' and GEMINI_AVAILABLE and ai_config.get('key'):
                try:
                    genai.configure(api_key=ai_config['key'])
                    self.ai_client = genai.GenerativeModel('gemini-pro')
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
    
    def get_actual_article_url(self, google_news_url):
        """Extract the real article URL from Google News redirect"""
        try:
            response = requests.get(google_news_url, headers=self.headers, timeout=10, allow_redirects=True)
            actual_url = response.url
            
            # If it's still a Google News URL, try to parse the redirect parameter
            if 'news.google.com' in actual_url:
                # Try to extract URL from the redirect
                match = re.search(r'url=([^&]+)', actual_url)
                if match:
                    from urllib.parse import unquote
                    return unquote(match.group(1))
            else:
                return actual_url
        except:
            pass
        
        return google_news_url
    
    def extract_article_content(self, url):
        """Extract article content from URL - PRIMARY METHOD"""
        
        if not url:
            return None
        
        # If it's a Google News URL, get the actual article URL first
        if 'news.google.com' in url:
            url = self.get_actual_article_url(url)
        
        # Skip if still Google News after redirect
        if 'news.google.com' in url:
            return None
        
        # Method 1: Trafilatura (BEST for news)
        if TRAFILATURA_AVAILABLE:
            try:
                downloaded = trafilatura.fetch_url(url, timeout=15)
                if downloaded:
                    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                    if text and len(text) > 150:
                        return text[:20000]
            except:
                pass
        
        # Method 2: Newspaper3k
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url, request_timeout=15)
                article.download()
                article.parse()
                
                if article.text and len(article.text) > 150:
                    return article.text[:20000]
            except:
                pass
        
        # Method 3: BeautifulSoup aggressive extraction
        try:
            response = requests.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements FIRST
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript', 'meta', 'link', 'button', 'form', 'ul', 'ol', 'li', 'svg', 'img']):
                element.decompose()
            
            # Try to find article content
            article_content = None
            
            # Try specific article selectors first
            selectors = [
                'article',
                'main',
                '[role="main"]',
                '.article-body',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.story-body',
                '.news-body',
                '[itemprop="articleBody"]',
                '.article-text',
                '.content-main',
                '.prose',
                '[role="article"]',
                '.full-content',
                '.article__body',
                '.article-wrapper',
                '.story-content'
            ]
            
            for selector in selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        article_content = element
                        break
                except:
                    pass
            
            # If no article element found, use body
            if not article_content:
                article_content = soup.find('body')
            
            if article_content:
                # Extract ONLY paragraph text - NO lists, NO navigation
                text_parts = []
                
                # Get all paragraphs from the content
                paragraphs = article_content.find_all('p', recursive=True)
                
                for para in paragraphs:
                    # Get clean text
                    text = para.get_text(separator=' ', strip=True)
                    
                    # Clean extra whitespace
                    text = re.sub(r'\s+', ' ', text).strip()
                    
                    # Only add substantial paragraphs (not menu items or short text)
                    if len(text) > 40 and len(text) < 1000:
                        # Skip if it's navigation/menu
                        skip_keywords = ['cookie', 'advertisement', 'subscribe', 'javascript', 'loading', 'follow us', 'share this', 'read more', 'sign in', 'log in', 'menu', 'navigation']
                        if not any(skip in text.lower() for skip in skip_keywords):
                            text_parts.append(text)
                
                # If no paragraphs found, try divs with text content
                if len(text_parts) < 3:
                    divs = article_content.find_all('div', recursive=True)
                    for div in divs[:30]:  # Limit to first 30 divs
                        # Skip if it contains lists
                        if div.find(['ul', 'ol', 'li', 'nav', 'header', 'footer']):
                            continue
                        
                        text = div.get_text(separator=' ', strip=True)
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        if len(text) > 40 and len(text) < 1000:
                            skip_keywords = ['cookie', 'advertisement', 'subscribe', 'javascript', 'loading']
                            if not any(skip in text.lower() for skip in skip_keywords):
                                if text not in text_parts:  # Avoid duplicates
                                    text_parts.append(text)
                
                if text_parts:
                    # Join paragraphs with newlines
                    full_text = '\n\n'.join(text_parts[:100])
                    
                    # Final cleanup
                    full_text = re.sub(r'\s+', ' ', full_text)
                    
                    if len(full_text) > 150:
                        return full_text[:20000]
        
        except Exception as e:
            pass
        
        return None
    
    def summarize_with_ai(self, text):
        """Summarize content using AI"""
        if not text or len(text) < 100 or not self.ai_client:
            return self.summarize_content(text)
        
        try:
            text_to_use = text[:5000]
            
            prompt = f"""Summarize this news article in 3-5 clear sentences:

{text_to_use}

Summary:"""
            
            if self.ai_type == 'gemini':
                response = self.ai_client.generate_content(prompt, timeout=10)
                return response.text.strip()
                
            elif self.ai_type == 'openai':
                response = self.ai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250,
                    timeout=10
                )
                return response.choices[0].message.content.strip()
                
            elif self.ai_type == 'anthropic':
                response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers={
                        'x-api-key': self.anthropic_key,
                        'anthropic-version': '2023-06-01',
                        'content-type': 'application/json'
                    },
                    json={
                        'model': 'claude-3-haiku-20240307',
                        'max_tokens': 250,
                        'messages': [{'role': 'user', 'content': prompt}]
                    },
                    timeout=10
                )
                return response.json()['content'][0]['text'].strip()
                
        except Exception as e:
            pass
        
        return self.summarize_content(text)
    
    def summarize_content(self, text, max_sentences=3):
        """Fallback simple summary"""
        if not text:
            return "No summary available"
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        summary = '. '.join(sentences[:max_sentences])
        return (summary + '.') if summary else "No summary available"
    
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
            clean = re.sub(r'&nbsp;|&amp;|&quot;|&apos;', '', clean)
            
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
                    
                    # Parse date
                    try:
                        from email.utils import parsedate_to_datetime
                        article_datetime = parsedate_to_datetime(pub_date)
                        article_datetime = article_datetime.replace(tzinfo=None)
                    except:
                        article_datetime = datetime.now()
                    
                    # Filter by date
                    if start_date and end_date:
                        if not (start_date <= article_datetime.date() <= end_date):
                            continue
                    
                    # UPDATE PROGRESS
                    if progress_callback:
                        progress_callback(idx + 1, len(items), f"🔗 Fetching: {title[:50]}...")
                    
                    # Get actual article URL (not Google News redirect)
                    actual_url = self.get_actual_article_url(article_url)
                    
                    # EXTRACT FULL ARTICLE CONTENT FROM ACTUAL URL
                    st.write(f"📥 Extracting from: {actual_url[:70]}...")
                    full_content = self.extract_article_content(actual_url)
                    
                    # FALLBACK to RSS description if extraction fails
                    if not full_content or len(full_content) < 150:
                        rss_desc = self.clean_rss_description(description_tag)
                        if rss_desc and len(rss_desc) > 80:
                            full_content = rss_desc
                    
                    # If still nothing, use title + some context
                    if not full_content or len(full_content) < 80:
                        full_content = f"{title}\n\nSource: {actual_url}"
                    
                    # GENERATE SUMMARY from extracted content
                    summary = self.summarize_with_ai(full_content) if len(full_content) > 80 else self.summarize_content(full_content)
                    
                    article_data = {
                        'keyword': keyword,
                        'title': title,
                        'url': actual_url,  # Use actual article URL, not Google redirect
                        'date_time': article_datetime.isoformat(),
                        'full_content': full_content,
                        'summary': summary
                    }
                    
                    self.articles.append(article_data)
                    
                    if progress_callback:
                        progress_callback(idx + 1, len(items), f"✅ Done: {title[:50]}...")
                    
                    time.sleep(2)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            st.error(f"Error: {e}")
    
    def search_newsapi(self, keyword, api_key, num_results=10, start_date=None, end_date=None, progress_callback=None):
        """Search using NewsAPI.org"""
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': keyword,
            'apiKey': api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': num_results
        }
        
        if start_date:
            params['from'] = start_date.strftime('%Y-%m-%d')
        if end_date:
            params['to'] = end_date.strftime('%Y-%m-%d')
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'ok':
                articles_list = data.get('articles', [])
                for idx, article in enumerate(articles_list):
                    article_url = article.get('url', '')
                    
                    if progress_callback:
                        progress_callback(idx + 1, len(articles_list), f"🔗 Fetching: {article.get('title', '')[:50]}...")
                    
                    # Extract full content from URL
                    st.write(f"📥 Extracting: {article_url[:70]}...")
                    full_content = self.extract_article_content(article_url)
                    
                    # Fallback to NewsAPI content
                    if not full_content or len(full_content) < 150:
                        fallback = article.get('content', article.get('description', ''))
                        if len(fallback) > 80:
                            full_content = fallback
                    
                    # If still nothing, use title
                    if not full_content or len(full_content) < 80:
                        full_content = f"{article.get('title', 'No title')}\n\nSource: {article_url}"
                    
                    # Generate summary
                    summary = self.summarize_with_ai(full_content) if len(full_content) > 80 else self.summarize_content(full_content)
                    
                    article_data = {
                        'keyword': keyword,
                        'title': article.get('title', 'No title'),
                        'url': article_url,
                        'date_time': article.get('publishedAt', ''),
                        'full_content': full_content,
                        'summary': summary
                    }
                    self.articles.append(article_data)
                    
                    if progress_callback:
                        progress_callback(idx + 1, len(articles_list), article_data['title'][:50])
                    
                    time.sleep(1)
            else:
                st.error(f"NewsAPI Error: {data.get('message', 'Unknown')}")
                
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
    st.markdown("*Extracts full article content & AI summaries*")
    st.markdown("---")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        method = st.radio(
            "Select Method:",
            ["Google News (Free - RSS)", "NewsAPI.org (API Key)"],
            help="Both will extract full article content from source URLs"
        )
        
        st.subheader("🔑 API Keys")
        
        news_api_key = None
        if "NewsAPI" in method:
            news_api_key = st.text_input("NewsAPI Key:", type="password")
        
        use_ai_summary = st.checkbox("🤖 Enable AI Summaries", value=True)
        ai_config = None
        
        if use_ai_summary:
            ai_provider = st.selectbox(
                "AI Provider:",
                ["OpenAI (GPT-3.5)", "Gemini (Google)", "Claude (Anthropic)"]
            )
            
            ai_key = st.text_input(f"{ai_provider.split()[0]} Key:", type="password")
            
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
        st.subheader("🔍 Search")
        keywords_input = st.text_area(
            "Keywords (one per line):",
            height=100,
            placeholder="AI\nTechnology\nNews"
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
        
        if "NewsAPI" in method and not news_api_key:
            st.error("Enter NewsAPI key!")
            return
        
        keywords = [k.strip() for k in keywords_input.split('\n') if k.strip()]
        scraper = NewsArticleScraper(ai_config=ai_config if use_ai_summary else None)
        
        overall_progress = st.progress(0)
        status_text = st.empty()
        
        for idx, keyword in enumerate(keywords):
            status_text.markdown(f"### 🔍 Keyword: **{keyword}**")
            
            keyword_progress = st.progress(0)
            
            def progress_callback(current, total, title):
                keyword_progress.progress(min(current / total, 0.99))
            
            start_dt = datetime.combine(start_date, datetime.min.time()) if use_date_range and start_date else None
            end_dt = datetime.combine(end_date, datetime.max.time()) if use_date_range and end_date else None
            
            if "NewsAPI" in method:
                scraper.search_newsapi(keyword, news_api_key, num_articles, start_dt, end_dt, progress_callback)
            else:
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
            st.metric("Keywords", df['Keyword'].nunique())
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
                st.markdown(f"**Keyword:** {row['Keyword']} | **Date:** {str(row['Date/Time'])[:10]}")
                st.markdown(f"[🔗 Read]({row['URL']})")
                st.info(f"**Summary:** {row['Summary']}")
                if st.checkbox("Full content", key=f"_{idx}"):
                    st.text_area("Content", row['Full Content'], height=250, disabled=True, key=f"c_{idx}")
        
        if st.button("🗑️ Clear"):
            st.session_state.scraped_data = None
            st.session_state.scraping_complete = False
            st.rerun()
    
    elif not st.session_state.scraping_complete:
        st.info("👈 Configure and click Scrape")


if __name__ == "__main__":
    main()