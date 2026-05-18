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
        # 1. Fetch the API key explicitly from system variables (Hugging Face Secrets)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        # Initialize Embeddings with the explicit key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key
        )
        
        # 2. Local Disk Persistence Connection Logic (Fixes Connection Refused errors)
        # We target a relative folder directory straight inside the Space application container
        CHROMA_DIR = "./chroma_db"
        print(f"Initializing Local Persistent ChromaDB storage at: {CHROMA_DIR}...")
        
        # Use PersistentClient instead of HttpClient for self-contained Spaces
        self.local_client = chromadb.PersistentClient(path=CHROMA_DIR)
        
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
        """Processes file and adds to the LOCAL Persistent ChromaDB Client."""
        docs = self.load_document(file_path)
        chunks = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks.")

        # Connect to the local collection engine via the PersistentClient
        vectorstore = Chroma(
            client=self.local_client,
            collection_name="research_assistant",
            embedding_function=self.embeddings
        )
        
        vectorstore.add_documents(chunks)
        print(f"Successfully indexed {file_path} directly to local disk persistence.")
        return vectorstore

if __name__ == "__main__":
    # Local fallback execution test
    test_file = os.path.join("data", "processed_data", "docs_clean.txt")
    
    if os.path.exists(test_file):
        pipeline = IngestionPipeline()
        pipeline.run(test_file)
    else:
        print(f"Error: {test_file} not found.")