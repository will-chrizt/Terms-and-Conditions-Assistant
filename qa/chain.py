from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever, task="qa"):
    """
    Builds a RetrievalQA chain for a given task.
    
    Args:
        llm: Your LLM instance (e.g., ChatBedrock)
        retriever: Vector store retriever (RAG)
        task: Type of task: "qa", "risk", "fairness", "hypothetical", "summary"
    Returns:
        RetrievalQA chain
    """
    # Define task-specific prompts
    prompts = {
        "qa": """You are a helpful legal assistant that explains Terms & Conditions in simple language. 
Always give clear, concise answers and highlight important rules that affect the user.

If the context does not contain the answer, say:
"I'm not able to find this information in the provided Terms & Conditions."

When answering:
- Use plain English (avoid legal jargon).
- Mention if the clause favors the company or the user.
- If possible, explain the risks for the user.

Context: {context}

Question: {question}

Answer:""",
        
        "risk": """You are analyzing Terms & Conditions to identify risky clauses. 
Classify each clause into High, Medium, or Low risk for the user.
Include a short explanation for each risk.

Context: {context}

Task: Identify risk levels for the user.

Answer:""",
        
        "fairness": """You are analyzing Terms & Conditions to find clauses that may be unfair or one-sided. 
Explain why each clause may be risky and provide a risk level.

Context: {context}

Task: Identify unfair clauses.

Answer:""",
        
        "hypothetical": """You are creating hypothetical scenarios where users might violate the Terms & Conditions. 
Provide 5 realistic scenarios with consequences.

Context: {context}

Task: Generate scenarios.

Answer:""",
        
        "summary": """Summarize the following Terms & Conditions in plain English.
Focus on key points: user rights, restrictions, payments, refunds, account termination, and data sharing.
Keep it concise.

Context: {context}

Answer:"""
    }

    prompt_template = PromptTemplate(
        template=prompts.get(task, prompts["qa"]),
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
