from langchain_ollama import ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts.chat import ChatPromptTemplate
from .config import LLM_MODEL_NAME

def build_rag_chain(retriever):
    """
    Constructs the RAG chain.
    """
    system_prompt = (
        "You are Lt. Commander Data from Star Trek: The Next Generation. "
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )
    
    return rag_chain

def query_chain(chain, question):
    result = chain.invoke({"input": question})
    return result["answer"]
