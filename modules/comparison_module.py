# modules/comparison_module.py
import streamlit as st
from langchain.prompts import PromptTemplate
from modules.summary_module import extract_summary_parameters   # ✅ reuse existing function
import json

import json
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from .summary_module import extract_summary_parameters
import re

def compare_documents(llm, doc1_text, doc2_text, top_k=5):
    """Compare two T&C documents dynamically, returning structured JSON with plain text Overall."""

    # Step 1: Detect parameters (reuse summary logic) → use Document objects
    params_doc1 = extract_summary_parameters(llm, [Document(page_content=doc1_text)], max_params=top_k)
    params_doc2 = extract_summary_parameters(llm, [Document(page_content=doc2_text)], max_params=top_k)

    # Step 2: Merge into unique set of parameters
    all_params = sorted(set(params_doc1 + params_doc2))[:top_k]

    comparisons = {}
    for param in all_params:
        prompt = PromptTemplate(
            input_variables=["doc1", "doc2", "param"],
            template="""
You are comparing two Terms & Conditions documents.

Focus on this parameter: {param}.

Return your answer strictly in **valid JSON** with this schema:
{{
  "Document A": "What Document A says about {param} (or 'Not explicitly stated')",
  "Document B": "What Document B says about {param} (or 'Not explicitly stated')",
  "Overall": "Plain text summary comparing both (no JSON, no code)"
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
            # ✅ Extract JSON part even if extra text is returned
            json_match = re.search(r"\{.*\}", response.content, re.DOTALL)
            if json_match:
                comparisons[param] = json.loads(json_match.group(0))
            else:
                raise ValueError("No JSON found in response")
        except Exception:
            comparisons[param] = {
                "Document A": "Error parsing response",
                "Document B": "Error parsing response",
                "Overall": response.content.strip()
            }

    return comparisons

def show_comparison(llm, doc1_text, doc2_text, top_k=5):
    """Streamlit UI for comparing two documents with expandable sections + clean formatting."""
    st.subheader("📊 Terms & Conditions Comparison")

    with st.spinner(f"Analyzing top {top_k} parameters..."):
        results = compare_documents(llm, doc1_text, doc2_text, top_k=top_k)

    for param, result in results.items():
        with st.expander(f"🔹 {param}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**📄 Document A**")
                st.write(result.get("Document A", "Not provided"))

            with col2:
                st.markdown("**📄 Document B**")
                st.write(result.get("Document B", "Not provided"))

            st.markdown("---")
            st.markdown("**🔍 Overall Comparison**")
            st.info(result.get("Overall", "No summary"))
