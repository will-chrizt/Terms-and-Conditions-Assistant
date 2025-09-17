import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from qa.chain import build_qa_chain

def show_risk_dashboard(llm, vector_store):
    if st.button("🔍 Analyze Risks in T&C"):
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        qa_chain = build_qa_chain(llm, retriever)

        risk_query = """
        Scan the T&C and classify important clauses into High, Medium, or Low risk for the user.
        Focus on refund terms, liability, account termination, data privacy, and fees.
        Output in a Markdown table with columns:
        | Clause (summarized) | Risk Level (High/Medium/Low) |
        """

        st.session_state["risk_analysis"] = qa_chain.invoke({"query": risk_query})["result"]

    if "risk_analysis" in st.session_state:
        st.markdown("### 📋 Risk Analysis Report")

        # Parse Markdown table → DataFrame
        lines = st.session_state["risk_analysis"].splitlines()
        rows = [line.strip("|").split("|") for line in lines if "|" in line][1:]  # skip header
        data = [{"Clause": r[0].strip(), "Risk Level": r[1].strip()} for r in rows if len(r) >= 2]
        df = pd.DataFrame(data)

        # Apply risk colors
        def highlight_risk(val):
            if val.lower() == "high":
                return "color: red; font-weight: bold;"
            elif val.lower() == "medium":
                return "color: orange; font-weight: bold;"
            elif val.lower() == "low":
                return "color: green; font-weight: bold;"
            return ""
        
        st.dataframe(df.style.applymap(highlight_risk, subset=["Risk Level"]))

        # Count risks
        high = (df["Risk Level"].str.lower() == "high").sum()
        medium = (df["Risk Level"].str.lower() == "medium").sum()
        low = (df["Risk Level"].str.lower() == "low").sum()

        st.markdown(f"""
        **Summary:**  
        - 🔴 High Risk: {high} clauses  
        - 🟠 Medium Risk: {medium} clauses  
        - 🟢 Low Risk: {low} clauses
        """)

        # Pie chart (doughnut style)
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            [high, medium, low],
            labels=["High", "Medium", "Low"],
            autopct='%1.1f%%',
            startangle=90,
            colors=["#ff4d4d", "#ffa64d", "#5cd65c"],
            textprops={'color': "white", 'weight': "bold"}
        )
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig.gca().add_artist(centre_circle)
        ax.axis("equal")

        st.pyplot(fig)
