import streamlit as st
from datetime import datetime
import pandas as pd
from io import BytesIO
from tavily import TavilyClient
import google.generativeai as genai


# Set page config
st.set_page_config(
    page_title="News Scraper",
    page_icon="📰",
    layout="wide"
)

def search_tavily(api_key, query, max_results=5, fetch_full_content=True, start_date=None, end_date=None):
    """Search using Tavily API"""
    try:
        tavily_client = TavilyClient(api_key=api_key)
        
        # Format dates if they are provided
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else None
        end_date_str = end_date.strftime('%Y-%m-%d') if end_date else None
        
        response = tavily_client.search(
            query=query,
            search_depth="advanced",
            topic="news",  # Required for date filtering
            include_answer=False,
            include_raw_content=fetch_full_content,
            max_results=max_results,
            start_date=start_date_str,
            end_date=end_date_str,
            include_domains=[],
            exclude_domains=[]
        )
        return response
    except Exception as e:
        st.error(f"Error calling Tavily API: {str(e)}")
        return None

def summarize_with_gemini(api_key, full_content):
    """Generate a short, bullet-point summary using the Gemini API."""
    if not full_content:
        return ""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = (
            "Based on the following article content, please generate a concise summary. "
            "The summary should be 2-3 bullet points and capture the main ideas of the text. "
            "Return only the bullet points.\n\n"
            f"ARTICLE CONTENT: \"{full_content}\""
        )
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.warning(f"Could not summarize with Gemini: {str(e)}")
        return "Summary could not be generated."

def summarize_content(content, max_length=300):
    """Create a bullet-point summary from the first few sentences."""
    if not content:
        return ""
    
    # A simple sentence tokenizer, splitting by period.
    # It filters out empty strings that may result from splitting.
    sentences = [s.strip() for s in content.replace('\n', ' ').split('.') if s.strip()]
    
    summary_bullets = []
    char_count = 0
    
    for sentence in sentences:
        # Add 3 for the bullet point formatting ("- " and a newline)
        if char_count + len(sentence) + 3 <= max_length:
            summary_bullets.append(f"- {sentence}.")
            char_count += len(sentence) + 3
        else:
            break
    
    # If no sentences could be added (e.g., first sentence is too long),
    # provide a truncated version of the first sentence as a fallback.
    if not summary_bullets and sentences:
        return f"- {sentences[0][:max_length-5]}..."

    return "\n".join(summary_bullets)

def process_results(results, gemini_api_key=None, use_gemini_summarizing=False):
    """Process Tavily results into structured data"""
    if not results or 'results' not in results:
        return []
    
    processed_data = []
    
    for item in results['results']:
        # Extract content
        title = item.get('title', 'N/A')
        url = item.get('url', 'N/A')
        content = item.get('content', '')
        raw_content = item.get('raw_content', '')
        
        # Use raw_content if available, otherwise use content
        full_content = raw_content if raw_content else content
        
        # Generate summary
        if use_gemini_summarizing and gemini_api_key:
            st.write(f"✨ Summarizing content for '{title}' with Gemini...")
            summary = summarize_with_gemini(gemini_api_key, full_content)
        else:
            summary = summarize_content(full_content)
        
        # Extract date (if available in the result)
        published_date = item.get('published_date', 'N/A')
        if published_date == 'N/A':
            published_date = datetime.now().strftime('%Y-%m-%d')
        
        processed_data.append({
            'Title': title,
            'Date': published_date,
            'URL': url,
            'Full Content': full_content,
            'Summary': summary,
            'Score': item.get('score', 0)
        })
    
    return processed_data

def create_excel_download(df):
    """Create Excel file for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='News Results')
        
        # Auto-adjust column widths
        worksheet = writer.sheets['News Results']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            # Cap at 50 for readability
            worksheet.column_dimensions[chr(65 + idx)].width = min(max_length + 2, 50)
    
    output.seek(0)
    return output

# Streamlit UI
st.title("📰News Scraper")
st.markdown("Search and scrape news articles with keyword queries (supports boolean operators)")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API keys from Streamlit secrets
    tavily_api_key = st.secrets.get("TAVILY_API_KEY")
    gemini_api_key = st.secrets.get("GEMINI_API_KEY")

    if not tavily_api_key:
        st.error("Tavily API key not found. Please add it to your secrets file.")

    max_results = st.slider(
        "Maximum Results",
        min_value=1,
        max_value=50,
        value=5,
        help="Number of results to fetch"
    )
    
    fetch_full_content = st.checkbox(
        "Fetch Full Content",
        value=True,
        help="Scrape the full content of articles. Disabling this can speed up searches."
    )

    use_gemini_summarizing = st.checkbox(
        "✨ Summarize with Gemini",
        value=False,
        help="Use Gemini API to generate a higher-quality summary. Requires a Gemini API key."
    )
    if use_gemini_summarizing and not gemini_api_key:
        st.warning("Gemini API key not found in secrets. Please add it to enable this feature.")

    st.markdown("---")
    st.header("📅 Date Range (Optional)")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None)
    with col2:
        end_date = st.date_input("End Date", value=None)
    
    st.markdown("---")
    st.markdown("### 💡 Boolean Query Tips")
    st.markdown("""
    - **AND**: `AI AND technology`
    - **OR**: `climate OR environment`
    - **NOT**: `tesla NOT stock`
    - **Quotes**: `"artificial intelligence"`
    - **Combine**: `(AI OR ML) AND ethics`
    """)

# Main content area
query = st.text_input(
    "🔍 Enter your search query",
    placeholder="e.g., AI AND (technology OR innovation) NOT crypto",
    help="Use boolean operators: AND, OR, NOT, and quotes for exact phrases"
)

col1, col2 = st.columns([1, 5])
with col1:
    search_button = st.button("🚀 Search", type="primary", use_container_width=True)

# Initialize session state
if 'results_data' not in st.session_state:
    st.session_state.results_data = None
if 'df' not in st.session_state:
    st.session_state.df = None

# Search functionality
if search_button:
    if not tavily_api_key:
        st.error("⚠️ Tavily API key is not configured. Please add it to your secrets file.")
    elif not query:
        st.error("⚠️ Please enter a search query")
    else:
        with st.spinner("🔎 Searching and scraping content..."):
            results = search_tavily(tavily_api_key, query, max_results, fetch_full_content, start_date, end_date)
            
            if results:
                processed_data = process_results(results, gemini_api_key, use_gemini_summarizing)
                
                if processed_data:
                    st.session_state.results_data = processed_data
                    st.session_state.df = pd.DataFrame(processed_data)
                    st.success(f"✅ Found {len(processed_data)} results!")
                else:
                    st.warning("No results found for your query")
            else:
                st.error("Failed to fetch results. Please check your API key and try again.")

# Display results
if st.session_state.results_data:
    st.markdown("---")
    st.subheader(f"📊 Results ({len(st.session_state.results_data)} articles)")
    
    # Download button
    excel_file = create_excel_download(st.session_state.df)
    st.download_button(
        label="📥 Download Excel",
        data=excel_file,
        file_name=f"news_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False
    )
    
    # Display results in expandable sections
    for idx, item in enumerate(st.session_state.results_data, 1):
        with st.expander(f"📄 {idx}. {item['Title']}", expanded=(idx == 1)):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**📅 Date:** {item['Date']}")
                st.markdown(f"**🔗 URL:** [{item['URL']}]({item['URL']})")
                st.markdown(f"**⭐ Relevance Score:** {item['Score']:.2f}")
            
            with col2:
                st.markdown("**📝 Summary:**")
                st.info(item['Summary'] if item['Summary'] else "No summary available")
            
            st.markdown("**📰 Full Content:**")
            st.text_area(
                "Content",
                value=item['Full Content'] if item['Full Content'] else "No content available",
                height=200,
                key=f"content_{idx}",
                label_visibility="collapsed"
            )
    
    # Show dataframe preview
    st.markdown("---")
    st.subheader("📊 Data Preview")
    display_df = st.session_state.df[['Title', 'Date', 'URL', 'Summary']].copy()
    st.dataframe(display_df, use_container_width=True, height=300)

else:
    # Show empty state
    st.info("👆 Enter a search query and click the search button to get started!")
    
    st.markdown("---")
    st.markdown("### 🎯 Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔍 Advanced Search**")
        st.markdown("Use boolean operators for precise queries")
    
    with col2:
        st.markdown("**📄 Full Content**")
        st.markdown("Scrape complete article text")
    
    with col3:
        st.markdown("**📊 Excel Export**")
        st.markdown("Download results with one click")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Made  ❤️by Maaz Manzoor"
    "</div>",
    unsafe_allow_html=True
)
