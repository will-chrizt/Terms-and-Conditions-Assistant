import os
os.environ["USER_AGENT"] = "LegalQAApp/1.0 (+https://yourdomain.com)"

import streamlit as st
from langchain_aws import BedrockEmbeddings, ChatBedrock
from config import AWS_REGION, EMBED_MODEL, LLM_MODEL
from loaders.url_loader import load_from_url
from vectorstore.store import get_vector_store

from modules.summary_module import show_summary
from modules.qa_module import show_qa
from modules.violations_module import show_hypothetical_violations
from modules.risk_module import show_risk_dashboard
from modules.comparison_module import show_comparison

# --------------------------
# Initialize embeddings + LLM (cached)
# --------------------------
@st.cache_resource
def get_embeddings():
    return BedrockEmbeddings(model_id=EMBED_MODEL, region_name=AWS_REGION)

@st.cache_resource
def get_llm():
    return ChatBedrock(model_id=LLM_MODEL, region_name=AWS_REGION)

embeddings = get_embeddings()
llm = get_llm()

# --------------------------
# Utility functions
# --------------------------
@st.cache_data(show_spinner="Loading and chunking document...")
def load_documents_from_url(url: str):
    docs = load_from_url(url)
    for d in docs:
        d.metadata["source"] = url
    return docs

# --------------------------
# Main app
# --------------------------
def main():
    st.title("📘 Terms & Conditions Analyzer")

    url = st.text_input("Enter the URL of a Terms & Conditions page")
    if not url:
        return

    # Load documents from first URL
    all_docs = load_documents_from_url(url)
    st.success(f"✅ Loaded {len(all_docs)} chunks from {url}")

    # Create vector store for RAG
    vector_store = get_vector_store(all_docs, embeddings, source=url)

    # --------------------------
    # 1. Summary
    # --------------------------
    show_summary(llm, all_docs)

    # --------------------------
    # 2. Q&A
    # --------------------------
    show_qa(llm, vector_store)

    # --------------------------
    # 3. Hypothetical Violations
    # --------------------------
    show_hypothetical_violations(llm, vector_store)

    # --------------------------
    # 4. Risk Rating Dashboard
    # --------------------------
    show_risk_dashboard(llm, vector_store)

    # --------------------------
    # 5. Compare with Second URL
    # --------------------------
    url_b = st.text_input("Enter the second T&C URL for comparison (optional)")

    if url_b:
        # First doc (already loaded earlier)
        doc1_text = "\n".join([d.page_content for d in all_docs])

        # Second doc (newly loaded)
        docs_b = load_documents_from_url(url_b)
        st.success(f"✅ Loaded {len(docs_b)} chunks from {url_b}")
        doc2_text = "\n".join([d.page_content for d in docs_b])

        # Pass plain text into comparison module
        show_comparison(llm, doc1_text, doc2_text)
 


if __name__ == "__main__":
    main()
