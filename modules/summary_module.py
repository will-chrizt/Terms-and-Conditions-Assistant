# modules/summary_module.py
import streamlit as st

def extract_summary_parameters(llm, all_docs, max_params=6):
    """
    Ask the LLM to detect the most important sections/parameters of a Terms & Conditions document.
    Returns a list of section names.
    """
    full_text = "\n".join([doc.page_content for doc in all_docs])
    prompt = f"""
You are an expert in analyzing Terms & Conditions documents. 
Identify the {max_params} most important sections or parameters that are critical for a user to know.
Return the result as a simple Python list of short section names.

Document:
{full_text}
"""
    response = llm.invoke(prompt).content
    try:
        parameters = eval(response)  # if model returns ["User Rights", "Refunds", ...]
    except:
        # fallback: split lines
        parameters = [p.strip("-• \n") for p in response.split("\n") if p.strip()][:max_params]
    return parameters


def summarize_terms(llm, all_docs):
    """Summarize the T&C using the entire document and AI-detected main parameters."""
    # Detect main parameters
    main_params = extract_summary_parameters(llm, all_docs)
    
    # Combine all document chunks
    full_text = "\n".join([doc.page_content for doc in all_docs])
    
    summary_prompt = f"""
You are an expert in reading legal Terms & Conditions documents. 
Summarize the following Terms & Conditions in plain English.
Focus on these main sections: {', '.join(main_params)}
Keep the summary concise and present it as bullet points.

Terms & Conditions:
{full_text}
"""
    summary_response = llm.invoke(summary_prompt).content
    return summary_response


def show_summary(llm, all_docs):
    """Display the summary in Streamlit."""
    with st.expander("📋 Summary of Terms & Conditions", expanded=True):
        summary = summarize_terms(llm, all_docs)
        st.write(summary)
