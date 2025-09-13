import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF for PDF highlighting

# Configure AWS Bedrock client
os.environ["AWS_REGION_NAME"] = "us-east-1"

# Initialize Bedrock embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)

# Initialize Claude model
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1"
)

# Function to load and process documents
def load_documents(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
    elif file_type == "txt":
        loader = TextLoader(file_path)
    else:
        return []
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

# Function to create or load vector store
def create_vector_store(docs, store_path="faiss_index"):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store

# Function to highlight text in PDF
def highlight_text_in_pdf(pdf_path, highlights, output_path="highlighted.pdf"):
    doc = fitz.open(pdf_path)
    for item in highlights:
        page_number = item["page"]
        text = item["text"]
        try:
            page = doc[page_number]
            text_instances = page.search_for(text)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.update()
        except Exception as e:
            print(f"Could not highlight: {text[:50]}... (Page {page_number}) | Error: {e}")
    doc.save(output_path)
    return output_path

# Streamlit app
def main():
    st.title("Legal Document Q&A System with Multi-file Support & Highlights")
    
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_docs = []
        temp_files = []
        
        for uploaded_file in uploaded_files:
            file_ext = uploaded_file.name.split(".")[-1].lower()
            temp_path = f"temp_{uploaded_file.name}"
            temp_files.append(temp_path)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            docs = load_documents(temp_path, file_ext)
            all_docs.extend(docs)
        
        st.write(f"Loaded {len(all_docs)} document chunks from {len(uploaded_files)} files")
        
        vector_store = create_vector_store(all_docs)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        prompt_template = PromptTemplate(
            template="""You are a legal assistant. Use the following context to answer the question accurately and concisely. 
If the context doesn't provide enough information, say so.

Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        query = st.text_input("Enter your question about the documents")
        if query:
            result = qa_chain.invoke({"query": query})
            
            st.write("**Answer:**")
            st.write(result["result"])
            
            highlights = []
            st.write("**Source Citations:**")
            for doc in result["source_documents"]:
                page_num = doc.metadata.get("page", 0)
                snippet = doc.page_content.strip()
                st.markdown(f"- Page {page_num+1}: `{snippet[:200]}...`")
                
                # Only try highlighting if it's a PDF
                if "pdf" in doc.metadata.get("source", "").lower():
                    highlights.append({"page": page_num, "text": snippet})
            
            # Highlight in first PDF only (demo)
            for temp_path in temp_files:
                if temp_path.endswith(".pdf") and highlights:
                    highlighted_pdf = highlight_text_in_pdf(temp_path, highlights)
                    with open(highlighted_pdf, "rb") as f:
                        st.download_button(
                            f"Download {os.path.basename(temp_path)} with Highlights",
                            f,
                            file_name=f"highlighted_{os.path.basename(temp_path)}",
                            mime="application/pdf"
                        )
        
        # Cleanup
        for temp in temp_files:
            os.remove(temp)

if __name__ == "__main__":
    main()
