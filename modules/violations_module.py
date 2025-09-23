import streamlit as st
import pandas as pd
from qa.chain import build_qa_chain

def show_hypothetical_violations(llm, vector_store):
    st.subheader("🚨 Hypothetical Policy Violations")
    
    if st.button("✨ Generate Scenarios"):
        # Retrieve relevant docs from the vector store
        retriever = vector_store.as_retriever(search_kwargs={"k": 15})
        docs = retriever.get_relevant_documents("terms")  # just fetch top docs

        chunks_text = "\n".join([doc.page_content for doc in docs])
        violation_prompt = """Based on the following Terms & Conditions, create 5 realistic hypothetical situations 
        where a user might unintentionally or intentionally break the policy. 
        Present the output in a Markdown table with the following columns:
        | Scenario | Violated Policy/Term | Possible Consequence |"""
        
        st.session_state["violations"] = llm.invoke(
            violation_prompt + "\n\n" + chunks_text
        ).content

    if "violations" in st.session_state:
        st.markdown("### 📋 Generated Scenarios")

        # Convert markdown table → DataFrame
        lines = st.session_state["violations"].splitlines()
        rows = [line.strip("|").split("|") for line in lines if "|" in line][1:]  # skip header
        data = []
        for r in rows:
            if len(r) >= 3:
                data.append({
                    "🚨 Scenario": r[0].strip(),
                    "⚖️ Violated Policy/Term": r[1].strip(),
                    "⚠️ Possible Consequence": r[2].strip()
                })
        df = pd.DataFrame(data)

        # Styled dataframe
        st.dataframe(
            df.style.set_properties(
                **{"white-space": "pre-wrap", "text-align": "left"}
            ).set_table_styles(
                [{"selector": "th", "props": [("font-weight", "bold"), ("background-color", "#f2f2f2")]}]
            )
        )

        # Export option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Scenarios as CSV",
            csv,
            "hypothetical_violations.csv",
            "text/csv",
            key="download-csv"
        )
