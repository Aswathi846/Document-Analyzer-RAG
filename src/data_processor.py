import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables safely
load_dotenv()

class GeminiRAG:
    def __init__(self):
        # 1. Fetch API Key from OS Environment (Supports Hugging Face Secrets natively)
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        # 2. Initialize Embeddings with the explicit key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        
        # 3. Dynamic Connection to the Shared Remote ChromaDB Client Engine
        CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
        CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
        
        print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
        remote_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

        # 4. Load Vector Store matching collection name from your ingestion pipeline
        self.vectorstore = Chroma(
            client=remote_client,
            collection_name="research_assistant",
            embedding_function=self.embeddings
        )
        
        # 5. Initialize the LLM with the explicit key
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.3,
            google_api_key=api_key
        )

    def retrieve(self, query, k=3):
        """Finds the most relevant technical snippets."""
        print(f"\n[RETRIEVAL] Searching for: '{query}'")
        return self.vectorstore.similarity_search(query, k=k)

    def generate(self, query, contexts):
        """Constructs the prompt and gets an answer from Gemini."""
        context_text = "\n\n---\n\n".join([doc.page_content for doc in contexts])
        
        prompt = f"""You are a technical AI expert. Answer the question using ONLY the provided research context.
        If the answer is not present, state that you do not have enough information.

        CONTEXT:
        {context_text}

        QUESTION: {query}

        ANSWER:"""
        
        print(f"[GENERATION] Consulting Gemini...")
        response = self.llm.invoke(prompt)

        # Robust type parsing for response structures
        if isinstance(response.content, list):
            if response.content:
                first_part = response.content[0]
                if isinstance(first_part, dict) and 'text' in first_part:
                    return first_part['text']
                return str(first_part)
            
        elif isinstance(response.content, str):
            return response.content

        # Safe fallback logic if string evaluations return empty configurations
        return getattr(response, 'content', str(response))

    def query_system(self, question):
        """The main orchestration flow."""
        relevant_docs = self.retrieve(question)
        answer = self.generate(question, relevant_docs)
        return answer

if __name__ == "__main__":
    # Test the system with a question about the 'Attention' paper
    rag = GeminiRAG()
    
    test_query = "What are the two main components of the Transformer architecture?"
    result = rag.query_system(test_query)
    
    print("\n" + "="*50)
    print(f"QUESTION: {test_query}")
    print(f"ANSWER:\n{result}")
    print("="*50)