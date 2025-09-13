import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Configure AWS Bedrock client (assumes AWS CLI credentials are set)
os.environ["AWS_REGION_NAME"] = "us-east-1"  # Adjust region as needed

# Initialize Bedrock embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    region_name="us-east-1"
)

# Initialize Claude model via Bedrock
llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    region_name="us-east-1"
)

# Function to load and process documents
def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(documents)

# Function to create or load vector store
def create_vector_store(docs, store_path="faiss_index"):
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(store_path)
    return vector_store

# Streamlit app
def main():
    st.title("Legal Document Q&A System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process document
        docs = load_documents("temp.pdf")
        st.write(f"Loaded {len(docs)} document chunks")
        
        # Create vector store
        vector_store = create_vector_store(docs)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Define prompt template
        prompt_template = PromptTemplate(
            template="""You are a legal assistant. Use the following context to answer the question accurately and concisely. If the context doesn't provide enough information, say so.

Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        # Set up RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # Query input
        query = st.text_input("Enter your question about the document")
        if query:
            result = qa_chain.invoke({"query": query})
            st.write("**Answer:**")
            st.write(result["result"])
            st.write("**Source Documents:**")
            for doc in result["source_documents"]:
                st.write(f"- {doc.page_content[:200]}... (Page {doc.metadata.get('page', 'N/A')})")
        
        # Clean up
        os.remove("temp.pdf")

if __name__ == "__main__":
    main()