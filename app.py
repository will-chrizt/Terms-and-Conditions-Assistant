import os
os.environ["USER_AGENT"] = "LegalQAApp/1.0 (+https://yourdomain.com)"

import streamlit as st
import matplotlib.pyplot as plt
from langchain_aws import BedrockEmbeddings, ChatBedrock
from config import AWS_REGION, EMBED_MODEL, LLM_MODEL
from loaders.file_loader import load_file
from loaders.url_loader import load_from_url
from vectorstore.store import create_vector_store
from qa.chain import build_qa_chain
from qa.highlight import highlight_text_in_pdf
from vectorstore.store import get_vector_store
 # old name, still works




# --------------------------
# Initialize embeddings + LLM (cached)
# --------------------------
@st.cache_resource
def get_embeddings():
    return BedrockEmbeddings(model_id=EMBED_MODEL, region_name=AWS_REGION)

@st.cache_resource
def get_llm():
    return ChatBedrock(model_id=LLM_MODEL, region_name=AWS_REGION)

embeddings = get_embeddings()
llm = get_llm()


# --------------------------
# Utility functions
# --------------------------
@st.cache_data(show_spinner="Loading and chunking document...")
def load_documents_from_url(url: str):
    docs = load_from_url(url)
    for d in docs:
        d.metadata["source"] = url
    return docs




def get_qa_chain(llm, retriever):
    return build_qa_chain(llm, retriever)


def extract_comparison_parameters(llm, all_docs):
    """Ask the LLM to suggest 5 parameters for comparison from the first document."""
    first_doc_text = "\n".join([doc.page_content for doc in all_docs][:20])
    param_prompt = f"""
You are analyzing a Terms & Conditions document. 
Based on the following text, identify 5 most important parameters/sections that are 
commonly compared across different policies.

Document:
{first_doc_text}

Return the result as a simple Python list of section names.
"""
    params_response = llm.invoke(param_prompt).content
    try:
        comparison_params = eval(params_response)  # if model returns ["Refunds", ...]
    except:
        # fallback: split lines
        comparison_params = [p.strip("-• \n") for p in params_response.split("\n") if p.strip()][:5]
    return comparison_params


# --------------------------
# Streamlit main app
# --------------------------
def main():
    st.title("📘 Terms & Conditions Analyzer")

    all_docs, sources = [], []  # track sources (files/urls)

    # --------------------------
    # 1. Enter ONE URL initially
    # --------------------------
    url = st.text_input("Enter the URL of a Terms & Conditions page")
    if url:
        docs = load_documents_from_url(url)
        all_docs.extend(docs)
        sources.append(url)
        st.success(f"✅ Loaded {len(all_docs)} chunks from {url}")

    # --------------------------
    # 2. Summarize + QA only after first URL
    # --------------------------
    if all_docs:
        with st.expander("📋 Summary of Terms & Conditions", expanded=True):
            @st.cache_data(show_spinner="Summarizing T&C...")
            def summarize_terms(llm, docs):
                summary_prompt = """Summarize the following Terms & Conditions in plain English.
                Focus on: user rights, restrictions, payments, refunds, account termination, and data sharing.
                Keep it under 10 bullet points."""
                chunks_text = "\n".join([doc.page_content for doc in docs[:15]])
                return llm.invoke(summary_prompt + "\n\n" + chunks_text).content

            summary = summarize_terms(llm, all_docs)
            st.write(summary)

        # Build retriever + chain (cached)
        vector_store = get_vector_store(all_docs, embeddings, source=url)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = get_qa_chain(llm, retriever)

        # --------------------------
        # 3. User Q&A
        # --------------------------
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
                    # Clickable source link
                    source_md = f"[Link]({source_url})" if source_url != "#" else "Unknown source"
        
                    with st.expander(f"Clause {i} (Source: {source_md}, Page {page})"):
                        st.write(clause_text)

        # --------------------------
        # 4. Hypothetical Violations
        # --------------------------
        st.subheader("🚨 Hypothetical Policy Violations")
        if st.button("Generate Scenarios"):
            violation_prompt = """Based on the following Terms & Conditions, create 5 realistic hypothetical situations 
            where a user might unintentionally or intentionally break the policy. 
            Present the output in a Markdown table with the following columns:
            | Scenario | Violated Policy/Term | Possible Consequence |"""
            chunks_text = "\n".join([doc.page_content for doc in all_docs[:15]])
            st.session_state["violations"] = llm.invoke(violation_prompt + "\n\n" + chunks_text).content

        if "violations" in st.session_state:
            st.markdown(st.session_state["violations"])

        # --------------------------
        # 5. Fairness Checker
        # --------------------------
        st.subheader("⚖️ Fairness Checker")
        if st.button("Check for Potentially Unfair Clauses"):
            fairness_prompt = """Analyze the following Terms & Conditions and identify clauses 
            that may be considered unfair, one-sided, or risky for the user.
            Present the findings in a Markdown table with columns:
            | Clause (summarized) | Why it may be unfair | Risk Level (High/Medium/Low) |"""
            chunks_text = "\n".join([doc.page_content for doc in all_docs[:15]])
            st.session_state["fairness"] = llm.invoke(fairness_prompt + "\n\n" + chunks_text).content

        if "fairness" in st.session_state:
            st.markdown(st.session_state["fairness"])

        # --------------------------
        # 6. Risk Rating Dashboard
        # --------------------------
       

        # --------------------------
        # 7. Compare with Second URL
        # --------------------------
        st.subheader("📑 Compare With Another Document")
        if st.button("Add Second URL for Comparison"):
            st.session_state["show_compare_input"] = True

        if st.session_state.get("show_compare_input"):
            second_url = st.text_input("Enter second T&C URL for comparison")
            if second_url:
                docs2 = load_documents_from_url(second_url)

                # Step 1: Get dynamic comparison parameters
                comparison_params = extract_comparison_parameters(llm, all_docs)

                # Step 2: Compare section-wise
                comparisons = {}
                for sec in comparison_params:
                    sec_texts = []
                    for src, docs in [(url, all_docs), (second_url, docs2)]:
                        src_docs = [doc.page_content for doc in docs]
                        src_text = "\n".join(src_docs[:20])
                        sec_texts.append(f"Source: {src}\n{src_text}")

                    section_prompt = f"""
Compare the following section: "{sec}" across the two documents and highlight key differences.

{"\n\n---\n\n".join(sec_texts)}

Present the output in a Markdown table with columns:
| Aspect | {url} | {second_url} | Key Difference Summary |
"""
                    comp = llm.invoke(section_prompt).content
                    comparisons[sec] = comp

                st.session_state["comparison_results"] = comparisons

        if "comparison_results" in st.session_state:
            for sec, comp in st.session_state["comparison_results"].items():
                with st.expander(f"📄 {sec} Differences", expanded=False):
                    st.markdown(comp)


if __name__ == "__main__":
    main()
