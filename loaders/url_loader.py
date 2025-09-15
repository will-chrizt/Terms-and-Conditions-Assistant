from langchain_community.document_transformers import Html2TextTransformer
from .file_loader import split_documents
from langchain.schema import Document

# Optional: only if PlaywrightURLLoader is still needed
from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup

def load_from_url(url: str):
    """
    Load text from a webpage (URL) with JS rendering (Playwright), then split into chunks.
    Falls back to requests + BeautifulSoup if Playwright fails.
    Returns a list of Document objects.
    """
    # --------------------------
    # 1. Try Playwright
    # --------------------------
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/135.0.0.0 Safari/537.36"
                )
            )
            page = context.new_page()
            page.goto(url, wait_until="load", timeout=30000)  # 30s timeout
            html_content = page.content()
            browser.close()

        # Convert HTML -> readable text
        doc = Document(page_content=html_content, metadata={"source": url})
        transformer = Html2TextTransformer()
        docs = transformer.transform_documents([doc])

        if docs and docs[0].page_content.strip():
            print(f"✅ Playwright extracted {sum(len(d.page_content) for d in docs)} chars from {url}")
            return split_documents(docs)

    except Exception as e:
        print(f"⚠️ Playwright failed for {url}: {e}")

    # --------------------------
    # 2. Fallback to requests + BeautifulSoup
    # --------------------------
    try:
        headers = {"User-Agent": "Mozilla/5.0 (LegalQAApp)"}
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator="\n").strip()
            if text:
                doc = Document(page_content=text, metadata={"source": url})
                print(f"✅ Fallback extracted {len(text)} chars from {url}")
                return split_documents([doc])
        print(f"⚠️ Fallback could not extract content from {url} (status {resp.status_code})")
    except Exception as e:
        print(f"⚠️ Requests fallback failed for {url}: {e}")

    # --------------------------
    # 3. Return empty list if all fails
    # --------------------------
    print(f"❌ No content extracted from {url}")
    return []
