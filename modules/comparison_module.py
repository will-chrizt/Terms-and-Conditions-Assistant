# modules/compare_module.py
import streamlit as st
from langchain.prompts import PromptTemplate

def detect_parameters(llm, doc_text):
    """Ask LLM to extract key clauses/parameters from a single document."""
    prompt = PromptTemplate(
        input_variables=["doc"],
        template="""
You are analyzing a Terms & Conditions document. 
Identify the main legal/contractual parameters (such as Refunds, Subscription, Privacy, Data Sharing, Account Termination, etc.).
Return them as a simple comma-separated list of unique parameters.

Document:
{doc}
"""
    )
    chain = prompt | llm
    response = chain.invoke({"doc": doc_text})
    return [p.strip() for p in response.content.split(",") if p.strip()]


def compare_documents(llm, doc1_text, doc2_text):
    """Compare two T&C documents dynamically with structured JSON output."""
    
    # Step 1: Detect parameters from each doc
    params_doc1 = detect_parameters(llm, doc1_text)
    params_doc2 = detect_parameters(llm, doc2_text)

    # Step 2: Merge into a unique set of parameters
    all_params = sorted(set(params_doc1 + params_doc2))

    # Step 3: Ask LLM to compare per parameter with JSON schema
    comparisons = {}
    for param in all_params:
        prompt = PromptTemplate(
            input_variables=["doc1", "doc2", "param"],
            template="""
You are comparing two Terms & Conditions documents.

Focus on this parameter: {param}.

Return the result strictly as JSON in this format:
{{
  "Document A": "Summary of what Document A says about {param} (or 'Not explicitly stated')",
  "Document B": "Summary of what Document B says about {param} (or 'Not explicitly stated')",
  "Overall": "Brief summary comparing both documents"
}}

Document A:
{doc1}

Document B:
{doc2}
"""
        )
        chain = prompt | llm
        response = chain.invoke({"doc1": doc1_text, "doc2": doc2_text, "param": param})

        try:
            import json
            comparisons[param] = json.loads(response.content)
        except Exception:
            # fallback if parsing fails
            comparisons[param] = {
                "Document A": "Error parsing response",
                "Document B": "Error parsing response",
                "Overall": response.content.strip()
            }

    return comparisons

def show_comparison(llm, doc1_text, doc2_text):
    """Streamlit UI for comparing two documents with expandable sections + table format."""
    st.subheader("📊 Terms & Conditions Comparison")

    with st.spinner("Analyzing and comparing documents..."):
        results = compare_documents(llm, doc1_text, doc2_text)

    for param, result in results.items():
        with st.expander(f"🔹 {param}", expanded=False):
            # Render clean table
            st.table(
                {
                    "Document A": [result.get("Document A", "Not provided")],
                    "Document B": [result.get("Document B", "Not provided")],
                }
            )

            # Overall summary
            st.markdown(f"**Overall:** {result.get('Overall', 'No summary')}")
