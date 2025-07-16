
import streamlit as st
from dotenv import load_dotenv
import os
from your_llms_generator import FirecrawlLLMsTextGenerator

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="llms.txt Generator | BossData", layout="wide")

st.title("🧠 llms.txt Generator | BossData")
st.markdown("""
Welcome to our **llms.txt Generator** — a tool to automatically summarize the content of a website using Firecrawl and OpenAI.

### 📘 What is llms.txt?
A `llms.txt` file is a structured summary of your website’s key pages, including titles and descriptions, intended to help large language models better understand and represent your content.

### ❓ Why create one?
- Improve visibility in AI-driven tools and chatbots
- Ensure accurate responses about your brand
- Highlight key pages and content types

### ⚙️ How it works
1. Enter your site URL below.
2. We'll crawl it using Firecrawl and summarize the content using OpenAI.
3. The number of crawled pages is 20 by default. However, you can choose more or fewer if you like. The maximum is 50 pages. 
4. You'll get a downloadable 'llms.txt' file.
5. The output shows how to best structure your llms.txt file. Use it as a starting point to improve and expand your own llms.txt file.
""")

url = st.text_input("Enter your website URL", placeholder="https://example.com")
max_urls = st.slider("Maximum number of pages to process", 5, 50, 20)

if st.button("Generate llms.txt"):
    if not url:
        st.warning("Please enter a URL.")
    elif not FIRECRAWL_API_KEY or not OPENAI_API_KEY:
        st.error("API keys are missing. Please configure your .env or Streamlit secrets.")
    else:
        with st.spinner("Processing..."):
            try:
                generator = FirecrawlLLMsTextGenerator(FIRECRAWL_API_KEY, OPENAI_API_KEY)
                result = generator.generate_llmstxt(url, max_urls)

                st.success(f"Success! Processed {result['num_urls_processed']} pages.")
                st.download_button(
                    label="📄 Download llms.txt",
                    data=result["llmstxt"],
                    file_name="llms.txt",
                    mime="text/plain"
                )

                st.markdown("### Preview")
                st.code(result["llmstxt"], language="markdown")
            except Exception as e:
                st.error(f"Something went wrong: {e}")