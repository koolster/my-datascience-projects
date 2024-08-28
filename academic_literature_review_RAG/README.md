# Academic Literature Review Chatbot

This project implements a chatbot that can answer questions about academic literature based on PDF documents. It uses LangChain, OpenAI, and ChromaDB to create a vector database of the documents and a chatbot interface.

## Setup

1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Set your OpenAI API key as an environment variable:
   - On Windows (Command Prompt):
     ```bash
     set OPENAI_API_KEY=your-api-key-here
     ```
   - On Windows (PowerShell):
     ```bash
     $env:OPENAI_API_KEY = "your-api-key-here"
     ```
   - On macOS/Linux:
     ```bash
     export OPENAI_API_KEY=your-api-key-here
     ```
   Replace `your-api-key-here` with your actual OpenAI API key.

## Usage

1. Place your PDF documents in the `pdf_documents` folder
2. Run the chatbot: `python main.py`
3. Open the provided URL in your web browser to interact with the chatbot

## Configuration

You can modify the `config/config.yaml` file to change settings such as the PDF folder path, chunk size for text splitting, and the model name.

