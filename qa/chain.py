from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def build_qa_chain(llm, retriever):
    prompt_template = PromptTemplate(
        template="""You are a legal assistant. Use the following context to answer the question accurately and concisely. 
If the context doesn't provide enough information, say so.

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

