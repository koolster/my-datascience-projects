from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

def setup_qa_chain(vector_store, model_name):
    """
    Set up the question-answering chain with the specified language model and vector store.
    """
    return RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(model_name=model_name),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

def generate_review(qa_chain, query):
    """
    Generate a review for the given query using the QA chain.
    """
    if qa_chain is None:
        return "Error: QA chain not initialized."
    result = qa_chain({"question": query})
    return f"Answer: {result['answer']}\n\nSources: {result['sources']}"
