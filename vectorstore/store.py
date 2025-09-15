from langchain_community.vectorstores import FAISS

def create_vector_store(docs, embeddings, store_path="faiss_index"):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store

def load_vector_store(embeddings, store_path="faiss_index"):
    return FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
