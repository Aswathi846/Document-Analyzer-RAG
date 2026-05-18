import os
import pandas as pd
import time
from src.pipeline import GeminiRAG
from dotenv import load_dotenv
from ragas import evaluate
# Use the individual metric classes for proper initialization
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset

load_dotenv()

def run_evaluation():
    # Initialize RAG system
    rag_system = GeminiRAG()
    
    # Initialize your judge LLM (Pro is often better for judging)
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest", # Recommended for higher accuracy judging
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0
    )

    test_set = [
        {
            "question": "What is the core benefit of the Transformer over RNNs?",
            "ground_truth": "The Transformer allows for significantly more parallelization and requires less time to train compared to recurrent models."
        },
        {
            "question": "Explain the Scaled Dot-Product Attention formula.",
            "ground_truth": "It computes the dot products of the query with all keys, divides by the square root of the key dimension, and applies a softmax function to the values."
        }
    ]

    results = []
    
    # Process each question to build the evaluation dataset
    for i, item in enumerate(test_set):
        print(f"Processing {i+1}/{len(test_set)}: {item['question']}")
        
        # Retrieval: Fetching the context chunks
        context_docs = rag_system.retriever.invoke(item['question'])
        contexts = [doc.page_content for doc in context_docs]

        # Generation: Getting the RAG system's answer
        answer = rag_system.query_system(item['question'])

        results.append({
            "question": item['question'],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item['ground_truth']
        })
        
        # Rate limit handling for Free Tier
        if i < len(test_set) - 1:
            print("Pausing 45 seconds to reset quota...")
            time.sleep(45)

    # Convert results list to a HuggingFace Dataset required by RAGAS
    dataset = Dataset.from_list(results)
    print("\n--- Running RAGAS Evaluation ---")
    
    try:
        # Initialize metrics as objects
        metrics = [
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ]

        # Final evaluation call
        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=rag_system.embeddings # Use the embeddings from your RAG system
        )
        
        df = score.to_pandas()
        print("\nEvaluation Results:")
        print(df[['question', 'faithfulness', 'answer_relevancy']])
        
        # Save report
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/evaluation_report.csv", index=False)
        print("\nReport saved to data/processed/evaluation_report.csv")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    run_evaluation()