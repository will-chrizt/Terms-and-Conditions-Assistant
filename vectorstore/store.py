from langchain_community.vectorstores import FAISS
import streamlit as st
import os, hashlib

def _hash_source(source: str) -> str:
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:8]

@st.cache_resource(show_spinner="🔍 Building or loading vector store...")
def get_vector_store(docs, embeddings, source: str, base_path="faiss_indexes"):
    os.makedirs(base_path, exist_ok=True)
    store_hash = _hash_source(source)
    store_path = os.path.join(base_path, f"faiss_{store_hash}")

    if os.path.exists(store_path):
        return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(store_path)
        return vector_store

# 👇 backward compatibility alias
create_vector_store = get_vector_store
