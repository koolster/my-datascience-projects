import os
import glob
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

def read_pdf(file_path):
    """
    Read a PDF file and extract its text content.
    """
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

class MetadataPreservingTextSplitter(RecursiveCharacterTextSplitter):
    """
    A text splitter that preserves metadata when splitting documents.
    """
    def split_documents(self, documents):
        texts = []
        for doc in documents:
            splits = self.split_text(doc.page_content)
            texts.extend([Document(page_content=split, metadata=doc.metadata) for split in splits])
        return texts

def load_documents(folder_path):
    """
    Load all PDF documents from a specified folder.
    """
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    documents = []
    for pdf_file in pdf_files:
        text = read_pdf(pdf_file)
        source = os.path.basename(pdf_file)
        doc = Document(page_content=text, metadata={"source": source})
        documents.append(doc)
    return documents

def create_vector_store(documents, chunk_size, chunk_overlap):
    """
    Create a vector store from the documents for efficient retrieval.
    """
    text_splitter = MetadataPreservingTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    return Chroma.from_documents(texts, embeddings)
