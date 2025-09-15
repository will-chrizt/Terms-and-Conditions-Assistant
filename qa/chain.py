from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    # T&C focused prompt
    prompt_template = PromptTemplate(
        template="""You are a helpful legal assistant that explains Terms & Conditions in simple language. 
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
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )
