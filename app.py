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
    input_choice = st.radio("Choose input type", ["Upload File(s)", "Enter URL(s)"])

    all_docs, temp_files, sources = [], [], []  # track sources (file names or URLs)

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
                sources.append(uploaded_file.name)

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                docs = load_file(temp_path, file_ext)
                for d in docs:
                    d.metadata["source"] = uploaded_file.name
                all_docs.extend(docs)

            st.success(f"✅ Loaded {len(all_docs)} chunks from {len(uploaded_files)} files")

    elif input_choice == "Enter URL(s)":
        urls = st.text_area("Enter one or more webpage URLs (one per line)")
        if urls:
            all_urls = [u.strip() for u in urls.splitlines() if u.strip()]
            for url in all_urls:
                docs = load_from_url(url)
                for d in docs:
                    d.metadata["source"] = url
                all_docs.extend(docs)
                sources.append(url)
            st.success(f"✅ Loaded {len(all_docs)} chunks from {len(all_urls)} URL(s)")

    # --------------------------
    # 2. Summarize T&C
    # --------------------------
    if all_docs:
        with st.expander("📋 Summary of Terms & Conditions", expanded=True):
            summary_prompt = """Summarize the following Terms & Conditions in plain English.
            Focus on: user rights, restrictions, payments, refunds, account termination, and data sharing.
            Keep it under 10 bullet points."""
            chunks_text = "\n".join([doc.page_content for doc in all_docs[:15]])  # take more chunks
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
                st.markdown(
                    f"- `{doc.page_content[:200]}...` (Source: {doc.metadata.get('source', '?')}, Page {doc.metadata.get('page', '?')})"
                )

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
        # 6. Hypothetical Violations
        # --------------------------
        st.subheader("🚨 Hypothetical Policy Violations")
        if st.button("Generate Hypothetical Scenarios"):
            violation_prompt = """Based on the following Terms & Conditions, create 5 realistic hypothetical situations 
            where a user might unintentionally or intentionally break the policy. 
            Present the output in a Markdown table with the following columns:
            | Scenario | Violated Policy/Term | Possible Consequence |
            Keep descriptions concise and clear."""
            chunks_text = "\n".join([doc.page_content for doc in all_docs[:15]])
            st.session_state["violations"] = llm.invoke(violation_prompt + "\n\n" + chunks_text).content

        if "violations" in st.session_state:
            st.markdown(st.session_state["violations"])

        # --------------------------
        # 7. Fairness Checker
        # --------------------------
        st.subheader("⚖️ Fairness Checker")
        if st.button("Check for Potentially Unfair Clauses"):
            fairness_prompt = """Analyze the following Terms & Conditions and identify clauses 
            that may be considered unfair, one-sided, or risky for the user.
            Present the findings in a Markdown table with columns:
            | Clause (summarized) | Why it may be unfair | Risk Level (High/Medium/Low) |
            Be concise and objective. Highlight clauses such as unilateral changes, no refunds, 
            excessive liability limitations, or forced arbitration."""
            chunks_text = "\n".join([doc.page_content for doc in all_docs[:15]])
            st.session_state["fairness"] = llm.invoke(fairness_prompt + "\n\n" + chunks_text).content

        if "fairness" in st.session_state:
            st.markdown(st.session_state["fairness"])

        # --------------------------
        # 8. Compare Multiple Documents (Highlight Differences)
        # --------------------------
        if len(sources) > 1:
            st.subheader("📑 Differences Between Documents")
            sections = [
                "Refund and cancellation policy",
                "Data collection and sharing",
                "Account termination conditions",
                "User obligations",
                "Fees or penalties"
            ]

            if st.button("Highlight Differences"):
                section_comparisons = {}
                for sec in sections:
                    sec_texts = []
                    for src in sources:
                        src_docs = [doc.page_content for doc in all_docs if doc.metadata.get("source") == src]
                        src_text = "\n".join(src_docs[:20])
                        sec_texts.append(f"Source: {src}\n{src_text}")

                    section_prompt = f"""
Compare the following section: "{sec}" across documents and highlight key differences.
Focus on policy wording and implications for the user.

{"\n\n---\n\n".join(sec_texts)}

Present the output in a Markdown table with columns:
| Aspect | {" | ".join(sources)} | Key Difference Summary |
"""
                    comparison_result = llm.invoke(section_prompt).content
                    section_comparisons[sec] = comparison_result

                st.session_state["comparison_by_section"] = section_comparisons

            if "comparison_by_section" in st.session_state:
                for sec, comp in st.session_state["comparison_by_section"].items():
                    with st.expander(f"📄 {sec} Differences", expanded=False):
                        st.markdown(comp)

    # --------------------------
    # Cleanup
    # --------------------------
    for temp in temp_files:
        os.remove(temp)

if __name__ == "__main__":
    main()
