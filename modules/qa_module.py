import streamlit as st
from qa.chain import build_qa_chain

def show_qa(llm, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = build_qa_chain(llm, retriever)

    query = st.text_input("🔎 Ask a question about this T&C")
    if query:
        result = qa_chain.invoke({"query": query})
        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Citations")
        if result.get("source_documents"):
            for i, doc in enumerate(result["source_documents"], start=1):
                source_url = doc.metadata.get("source", "#")
                page = doc.metadata.get("page", "?")
                clause_text = doc.page_content.strip().replace("\n", " ")
                source_md = f"[Link]({source_url})" if source_url != "#" else "Unknown source"

                with st.expander(f"Clause {i} (Source: {source_md}, Page {page})"):
                    st.write(clause_text)
