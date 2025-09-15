from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_file(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    else:
        return []

    documents = loader.load()
    return split_documents(documents)

def split_documents(documents, chunk_size=1500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_documents(documents)

