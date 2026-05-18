import os
import shutil
import chromadb
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from pathlib import Path

# Security: Load from hidden volume if in Docker, otherwise local
ENV_PATH = Path("/app/.env")
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None)

class IngestionPipeline:
    def __init__(self, model_name="models/gemini-embedding-001"):
        # 1. Initialize Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        
        # 2. Dynamic Connection Logic (Fixes the ValueError)
        IS_DOCKER = os.path.exists('/.dockerenv')

        CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
        CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))

        print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
        self.remote_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # 3. Setup Text Splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path):
        """Loads document based on file extension."""
        print(f"--- Loading: {file_path} ---")
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path}")
        
        return loader.load()

    def run(self, file_path):
        """Processes file and adds to the REMOTE ChromaDB Server."""
        docs = self.load_document(file_path)
        chunks = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks.")

        # Connect to the remote collection via the HttpClient
        vectorstore = Chroma(
            client=self.remote_client,
            collection_name="research_assistant",
            embedding_function=self.embeddings
        )
        
        vectorstore.add_documents(chunks)
        print(f"Successfully indexed {file_path} to the persistent server.")
        return vectorstore

if __name__ == "__main__":
    # Local testing: Ensure your Docker containers are RUNNING before executing this
    test_file = os.path.join("data", "processed_data", "docs_clean.txt")
    
    if os.path.exists(test_file):
        pipeline = IngestionPipeline()
        pipeline.run(test_file)
    else:
        print(f"Error: {test_file} not found.")