import streamlit as st
import matplotlib.pyplot as plt
from qa.chain import build_qa_chain

def show_risk_dashboard(llm, vector_store):
    if st.button("Analyze Risks in T&C"):
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = build_qa_chain(llm, retriever)

        risk_query = """Scan the T&C and classify important clauses into High, Medium, or Low risk for the user.
        Focus on refund terms, liability, account termination, data privacy, and fees.
        Output in a Markdown table with columns:
        | Clause (summarized) | Risk Level (High/Medium/Low) |"""

        st.session_state["risk_analysis"] = qa_chain.invoke({"query": risk_query})["result"]

    if "risk_analysis" in st.session_state:
        st.markdown(st.session_state["risk_analysis"])

        # Pie chart visualization
        high = st.session_state["risk_analysis"].lower().count("high")
        medium = st.session_state["risk_analysis"].lower().count("medium")
        low = st.session_state["risk_analysis"].lower().count("low")

        fig, ax = plt.subplots()
        ax.pie([high, medium, low], labels=["High Risk", "Medium Risk", "Low Risk"], autopct='%1.1f%%', startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
