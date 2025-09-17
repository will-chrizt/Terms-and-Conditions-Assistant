import streamlit as st
from qa.chain import build_qa_chain

def show_qa(llm, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    qa_chain = build_qa_chain(llm, retriever)

    st.markdown("## 🔎 Ask Questions about this T&C")

    # Centered input box with a placeholder
    query = st.text_input(
        "Type your question below:",
        placeholder="e.g., What data does this company collect?"
    )

    if query:
        with st.spinner("Thinking... 💭"):
            result = qa_chain.invoke({"query": query})

        # ✅ Display Answer
        st.markdown("### ✅ Answer")
        st.success(result["result"])

        # ✅ Display Citations in clean expandable cards
        if result.get("source_documents"):
            st.markdown("### 📖 Supporting Clauses & Sources")
            for i, doc in enumerate(result["source_documents"], start=1):
                source_url = doc.metadata.get("source", "#")
                page = doc.metadata.get("page", "?")
                clause_text = doc.page_content.strip().replace("\n", " ")

                # Build source markdown
                source_md = f"[🌐 Source Link]({source_url})" if source_url != "#" else "❓ Unknown source"

                with st.expander(f"📄 Clause {i} — Page {page}"):
                    st.markdown(f"**Source:** {source_md}")
                    st.markdown(f"**Excerpt:**\n\n> {clause_text}")
