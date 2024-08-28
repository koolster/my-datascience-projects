import gradio as gr
from .document_processor import load_documents, create_vector_store
from .qa_chain import setup_qa_chain, generate_review

def run_chatbot(config):
    """
    Run the chatbot with the given configuration.
    """
    # Load documents and create vector store
    documents = load_documents(config['pdf_folder'])
    vector_store = create_vector_store(documents, config['chunk_size'], config['chunk_overlap'])
    
    # Set up the QA chain
    qa_chain = setup_qa_chain(vector_store, config['model_name'])

    def respond(message):
        """
        Generate a response for the given message.
        """
        return generate_review(qa_chain, message)

    # Set up Gradio interface
    iface = gr.Interface(
        fn=respond,
        inputs="text",
        outputs="text",
        title="Literature Review Chatbot with References",
        description="Ask questions about the literature in the PDF documents. The response will include references to the source documents."
    )
    
    # Launch the interface
    url = iface.launch(share=True, server_port=config['server_port'])
    print(f"Gradio interface is running at: {url}")

    # Keep the script running
    gr.close_all()
    iface.launch()
