from langchain_community.document_loaders import WebBaseLoader
from .file_loader import split_documents

def load_from_url(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    return split_documents(documents)
