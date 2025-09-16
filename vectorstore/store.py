from langchain_community.vectorstores import FAISS
import streamlit as st
import os
import hashlib

def _hash_source(source: str) -> str:
    """Create a short hash of the source URL or identifier for caching."""
    return hashlib.md5(source.encode("utf-8")).hexdigest()[:8]

@st.cache_resource(show_spinner="🔍 Building or loading vector store...")
def get_vector_store(docs, embeddings, source: str, base_path="faiss_indexes"):
    """
    Build or load a FAISS vector store from documents and embeddings.
    Filters out empty documents to prevent FAISS errors.
    """
    os.makedirs(base_path, exist_ok=True)
    store_hash = _hash_source(source)
    store_path = os.path.join(base_path, f"faiss_{store_hash}")

    # Filter out empty chunks
    docs = [d for d in docs if d.page_content.strip()]
    if not docs:
        st.error(f"No valid text found to create vector store for {source}.")
        return None

    # Load existing FAISS index if available
    if os.path.exists(store_path):
        try:
            vector_store = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True
            )
            return vector_store
        except Exception as e:
            st.warning(f"Failed to load existing vector store, rebuilding: {e}")

    # Create new FAISS vector store
    try:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(store_path)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# Backward compatibility alias
create_vector_store = get_vector_store
