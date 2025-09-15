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
    st.title("📘 Terms & Conditions Q&A Assistant")
    input_choice = st.radio("Choose input type", ["Upload File(s)", "Enter URL"])

    all_docs, temp_files = [], []

    # --------------------------
    # 1. Load Documents
    # --------------------------
    if input_choice == "Upload File(s)":
        uploaded_files = st.file_uploader(
            "Upload T&C documents (PDF, DOCX, TXT)", 
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

            st.success(f"✅ Loaded {len(all_docs)} chunks from {len(uploaded_files)} files")

    elif input_choice == "Enter URL":
        url = st.text_input("Enter webpage URL")
        if url:
            docs = load_from_url(url)
            all_docs.extend(docs)
            st.success(f"✅ Loaded {len(all_docs)} chunks from {url}")

    # --------------------------
    # 2. Summarize T&C
    # --------------------------
    if all_docs:
        with st.expander("📋 Summary of Terms & Conditions", expanded=True):
            summary_prompt = """Summarize the following Terms & Conditions in plain English.
            Focus on: user rights, restrictions, payments, refunds, account termination, and data sharing.
            Keep it under 10 bullet points."""

            chunks_text = "\n".join([doc.page_content for doc in all_docs[:5]])  # first few chunks
            summary = llm.invoke(summary_prompt + "\n\n" + chunks_text)
            st.write(summary.content)

        # --------------------------
        # 3. Create Vector Store + QA Chain
        # --------------------------
        vector_store = create_vector_store(all_docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = build_qa_chain(llm, retriever)

        # --------------------------
        # 4. User Q&A
        # --------------------------
        query = st.text_input("🔎 Ask a question about these T&Cs")
        if query:
            result = qa_chain.invoke({"query": query})
            st.subheader("Answer")
            st.write(result["result"])

            st.subheader("Citations")
            for doc in result["source_documents"]:
                st.markdown(f"- `{doc.page_content[:200]}...` (Page {doc.metadata.get('page', '?')})")

            # Highlight important terms in PDFs
            keywords = ["refund", "liability", "terminate", "data", "privacy", "fees"]
            if temp_files:  
                pdf_path = temp_files[0]  # highlight only first uploaded PDF
                highlight_text_in_pdf(pdf_path, keywords)
                st.download_button("📥 Download highlighted T&C", "highlighted.pdf")

        # --------------------------
        # 5. Preset Quick Questions
        # --------------------------
        st.subheader("⚡ Quick Questions")
        preset_qs = [
            "Can I get a refund if I cancel?",
            "Can the company terminate my account without warning?",
            "Do they share my data with third parties?",
            "What are my responsibilities as a user?",
        ]
        for q in preset_qs:
            if st.button(q):
                result = qa_chain.invoke({"query": q})
                st.write(f"**Q:** {q}")
                st.write(f"**A:** {result['result']}")

    # --------------------------
    # Cleanup
    # --------------------------
    for temp in temp_files:
        os.remove(temp)

if __name__ == "__main__":
    main()
