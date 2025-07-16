
import os
import json
import time
import re
import logging
from urllib.parse import urlparse
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from openai import OpenAI

EXCLUDE_SEGMENTS = [
    "", "index", "home", "homepage", "privacy", "terms", "legal", "sitemap",
    "sitemap.xml", "robots.txt", "author", "authors", "admin", "login",
    "user-data", "settings", "internal-docs", "pricing", "strategy",
    "sales-materials", "confidential", "beta", "staging", "dev", "404",
    "search", "thank-you", "cart", "tag", "category", "archive"
]

class FirecrawlLLMsTextGenerator:
    def __init__(self, firecrawl_api_key: str, openai_api_key: str):
        self.firecrawl_api_key = firecrawl_api_key
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.firecrawl_base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.firecrawl_api_key}",
            "Content-Type": "application/json"
        }

    def _get_section_from_path(self, path: str) -> str:
        parts = path.strip("/").split("/")
        for part in parts:
            if part.lower() not in EXCLUDE_SEGMENTS:
                return part.capitalize()
        return "Misc"

    def map_website(self, url: str, limit: int = 100) -> List[str]:
        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/map",
                headers=self.headers,
                json={ "url": url, "limit": limit },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data.get("links", []) if data.get("success") else []
        except Exception as e:
            logging.error(f"Mapping error: {e}")
            return []

    def scrape_url(self, url: str) -> Optional[str]:
        try:
            response = requests.post(
                f"{self.firecrawl_base_url}/scrape",
                headers=self.headers,
                json={ "url": url, "formats": ["markdown"], "onlyMainContent": True },
                timeout=30
            )
            response.raise_for_status()
            data = response.json().get("data", {})
            return data.get("markdown", "")
        except Exception as e:
            logging.error(f"Scrape error: {e}")
            return None

    def generate_page_description(self, url: str, markdown: str) -> Tuple[str, str]:
        prompt = f"""Generate a 9-10 word description and a 3-4 word title of the entire page based on ALL the content one will find on the page for this url: {url}. This will help in a user finding the page for its intended purpose.

Return the response in JSON format:
{{
  "title": "3-4 word title",
  "description": "9-10 word description"
}}"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    { "role": "system", "content": "You are a helpful assistant that generates concise titles and descriptions for web pages." },
                    { "role": "user", "content": f"{prompt}\n\nPage content:\n{markdown[:4000]}" }
                ],
                temperature=0.3,
                max_tokens=100,
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("title", "Page"), result.get("description", "No description available")
        except Exception as e:
            logging.debug(f"OpenAI error for {url}: {e}")
            return "Page", "No description available"

    def generate_site_summary(self, markdown_pages: List[str]) -> Tuple[str, str]:
        combined_content = "\n\n".join(markdown_pages)[:8000]
        prompt = """Write a high-level name and a short summary of this organization and its online content. Use maximum 50 words. Return JSON format:
{
  "name": "...",
  "summary": "..."
}"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    { "role": "system", "content": "You are a helpful assistant that summarizes websites." },
                    { "role": "user", "content": f"{prompt}\n\nWebsite content:\n{combined_content}" }
                ],
                temperature=0.3,
                max_tokens=150,
                response_format={ "type": "json_object" }
            )
            result = json.loads(response.choices[0].message.content)
            return result.get("name", "Website"), result.get("summary", "This site contains multiple sections of informative content.")
        except Exception as e:
            logging.debug(f"OpenAI summary error: {e}")
            return "Website", "This site contains multiple sections of informative content."

    def generate_llmstxt(self, url: str, max_urls: int = 20) -> Dict[str, str]:
        all_urls = self.map_website(url, limit=max_urls)
        filtered_urls = []
        for u in all_urls:
            path_segments = urlparse(u).path.strip("/").split("/")
            if not any(seg.lower() in EXCLUDE_SEGMENTS for seg in path_segments):
                filtered_urls.append(u)

        pages = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.scrape_url, u): u for u in filtered_urls}
            for future in as_completed(futures):
                content = future.result()
                print(f"Scraping {futures[future]} =>", "OK" if content else "None")
                print(f"Scraping {futures[future]} =>", len(content) if content else "None")
                if content:
                    pages.append({ "url": futures[future], "markdown": content })

        sections: Dict[str, List[Tuple[str, str, str]]] = {}
        for page in pages:
            title, description = self.generate_page_description(page["url"], page["markdown"])
            path = urlparse(page["url"]).path
            section = self._get_section_from_path(path)
            sections.setdefault(section, []).append((title, description, page["url"]))

        site_name, summary = self.generate_site_summary([p["markdown"] for p in pages])

        llmstxt = f"# {site_name}\n\n{summary}\n\n"
        for section, items in sorted(sections.items()):
            llmstxt += f"## {section}\n\n"
            for title, desc, link in items:
                llmstxt += f"- [{title}]({link}): {desc}\n"
            llmstxt += "\n"

        return {
            "llmstxt": llmstxt.strip(),
            "num_urls_processed": len(pages),
            "num_urls_total": len(filtered_urls)
        }