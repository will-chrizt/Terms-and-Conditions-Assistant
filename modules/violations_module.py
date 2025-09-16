import streamlit as st
from qa.chain import build_qa_chain

def show_hypothetical_violations(llm, vector_store):
    st.subheader("🚨 Hypothetical Policy Violations")
    
    if st.button("Generate Scenarios"):
        # Retrieve relevant docs from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.get_relevant_documents("dummy")  # query doesn't matter, just fetch top docs

        chunks_text = "\n".join([doc.page_content for doc in docs])
        violation_prompt = """Based on the following Terms & Conditions, create 5 realistic hypothetical situations 
        where a user might unintentionally or intentionally break the policy. 
        Present the output in a Markdown table with the following columns:
        | Scenario | Violated Policy/Term | Possible Consequence |"""
        
        st.session_state["violations"] = llm.invoke(violation_prompt + "\n\n" + chunks_text).content

    if "violations" in st.session_state:
        st.markdown(st.session_state["violations"])
