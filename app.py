import os
os.environ["USER_AGENT"] = "LegalQAApp/1.0 (+https://yourdomain.com)"
import streamlit as st
from langchain_aws import BedrockEmbeddings, ChatBedrock
from config import AWS_REGION, EMBED_MODEL, LLM_MODEL
from loaders.file_loader import load_file
from loaders.url_loader import load_from_url
from vectorstore.store import create_vector_store
from qa.chain import build_qa_chain
from qa.highlight import highlight_text_in_pdf

# Initialize embeddings + LLM
embeddings = BedrockEmbeddings(model_id=EMBED_MODEL, region_name=AWS_REGION)
llm = ChatBedrock(model_id=LLM_MODEL, region_name=AWS_REGION)

def main():
    st.title("📘 Legal Document Q&A System")
    input_choice = st.radio("Choose input type", ["Upload File(s)", "Enter URL"])

    all_docs, temp_files = [], []

    if input_choice == "Upload File(s)":
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT)", 
            type=["pdf", "docx", "txt"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_ext = uploaded_file.name.split(".")[-1].lower()
                temp_path = f"temp_{uploaded_file.name}"
                temp_files.append(temp_path)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                docs = load_file(temp_path, file_ext)
                all_docs.extend(docs)

            st.write(f"Loaded {len(all_docs)} chunks from {len(uploaded_files)} files")

    elif input_choice == "Enter URL":
        url = st.text_input("Enter webpage URL")
        if url:
            docs = load_from_url(url)
            all_docs.extend(docs)
            st.write(f"Loaded {len(all_docs)} chunks from {url}")

    if all_docs:
        vector_store = create_vector_store(all_docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        qa_chain = build_qa_chain(llm, retriever)

        query = st.text_input("Enter your question")
        if query:
            result = qa_chain.invoke({"query": query})
            st.write("**Answer:**", result["result"])

            st.write("**Citations:**")
            for doc in result["source_documents"]:
                st.markdown(f"- `{doc.page_content[:200]}...` (Page {doc.metadata.get('page', '?')})")

    # Cleanup temp files
    for temp in temp_files:
        os.remove(temp)

if __name__ == "__main__":
    main()
