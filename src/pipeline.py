import os
import chromadb
from dotenv import load_dotenv
from pathlib import Path
from langsmith import traceable  # Added for LangSmith

# LangChain Imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Robust Import Strategy for Reranking
try:
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
except ImportError:
    try:
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        ContextualCompressionRetriever = None

try:
    from langchain_community.document_compressors.flashrank import FlashrankRerank
except ImportError:
    FlashrankRerank = None

ENV_PATH = Path("/app/.env")
load_dotenv(dotenv_path=ENV_PATH if ENV_PATH.exists() else None)

class GeminiRAG:
    def __init__(self):
        # 1. Fetch the API key explicitly from system variables (Hugging Face Secrets)
        #    or fallback to whatever dotenv grabbed locally.
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        # 2. Initialize Embeddings with explicit credential token
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        
        IS_DOCKER = os.path.exists('/.dockerenv')
        CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
        CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
        
        print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
        remote_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

        # 3. Load Vector Store
        self.vectorstore = Chroma(
            client=remote_client,
            collection_name="research_assistant",
            embedding_function=self.embeddings
        )
        
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        # 4. Initialize Reranker (Two-Stage Retrieval)
        if FlashrankRerank and ContextualCompressionRetriever:
            try:
                compressor = FlashrankRerank()
                self.retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
                print("System initialized with FlashRank Reranker.")
            except Exception as e:
                print(f"Reranker init failed: {e}. Falling back to base.")
                self.retriever = base_retriever
        else:
            self.retriever = base_retriever
            print("System initialized with Base Retriever.")

        # 5. Initialize LLM with explicit credential token
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest", 
            temperature=0.3,
            google_api_key=api_key
        )

    @traceable(run_type="retriever") # 2nd step: Captures retrieval & reranking
    def retrieve_and_rerank(self, question):
        """Used by Streamlit to get docs for the 'Evidence' expander."""
        print(f"[RETRIEVAL] Fetching candidates for: '{question}'")
        return self.retriever.invoke(question)

    @traceable(run_type="llm") # 3rd step: Captures specific generation latency
    def generate(self, question, relevant_docs):
        """Core generation logic using a technical persona."""
        context_text = "\n\n---\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = f"""You are a technical AI expert. Answer the question using ONLY the provided context.
        If the answer is not in the context, say you don't have enough information.

        CONTEXT:
        {context_text}

        QUESTION: {question}

        ANSWER:"""

        print(f"[GENERATION] Consulting Gemini...")
        response = self.llm.invoke(prompt)
        
        # Robust content parsing
        content = response.content
        if isinstance(content, list):
            return " ".join([part['text'] if isinstance(part, dict) and 'text' in part else str(part) for part in content])
        return content

    @traceable(run_type="chain") # 1st step: The top-level parent trace
    def query_system(self, question):
        """Standard orchestration for the terminal/CLI mode."""
        docs = self.retrieve_and_rerank(question)
        return self.generate(question, docs)

if __name__ == "__main__":
    rag = GeminiRAG()
    print("\nREADY: Ask about 'Attention Is All You Need'")
    
    while True:
        user_query = input("\n[USER]: ")
        if user_query.lower() in ['exit', 'quit', 'q']:
            break
        if not user_query.strip():
            continue
            
        result = rag.query_system(user_query)
        print(f"\n[AI]: {result}")