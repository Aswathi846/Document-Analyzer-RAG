import os
import pandas as pd
import time
import json
from datetime import datetime
from src.pipeline import GeminiRAG
from dotenv import load_dotenv
from langchain_core.tracers.context import collect_runs

# --- CRITICAL RAGAS WRAPPER IMPORTS ---
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset

load_dotenv()

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

def run_evaluation():
    # Initialize RAG system
    rag_system = GeminiRAG()
    
    # 1. Properly Wrap the Judge LLM with LangchainLLMWrapper
    raw_judge_llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",  # High-speed flash model recommended for evaluations
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0
    )
    judge_llm = LangchainLLMWrapper(raw_judge_llm)

    # 2. Properly Wrap Embeddings with LangchainEmbeddingsWrapper
    judge_embeddings = LangchainEmbeddingsWrapper(rag_system.embeddings)

    results = []
    
    for i, item in enumerate(test_set):
        print(f"Processing {i+1}/{len(test_set)}: {item['question']}")
        
        # Retrieval
        relevant_docs = rag_system.retrieve_and_rerank(item['question'])
        contexts = [doc.page_content for doc in relevant_docs]

        # Generation
        answer = rag_system.generate(item['question'], relevant_docs)

        results.append({
            "question": item['question'],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item['ground_truth']
        })
        
        if i < len(test_set) - 1:
            print("Pausing to respect rate limits...")
            time.sleep(15) 

    dataset = Dataset.from_list(results)
    print("\n--- Running RAGAS Evaluation ---")
    
    with collect_runs() as cb:
        try:
            metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

            # Execute evaluation passing our wrapped models
            score = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=judge_embeddings
            )
            
            run_id = cb.traced_runs[0].id if cb.traced_runs else None
            print(f"Traced to LangSmith! Run ID: {run_id}")
            
            df = score.to_pandas()
            
            summary_metrics = {
                "faithfulness": float(df['faithfulness'].mean()) if 'faithfulness' in df else 0.0,
                "relevancy": float(df['answer_relevancy'].mean()) if 'answer_relevancy' in df else 0.0,
                "precision": float(df['context_precision'].mean()) if 'context_precision' in df else 0.0,
                "total_tests": len(test_set),
                "last_run": datetime.now().strftime("%Y-%m-%d %H:%M")
            }

            # --- MATCHES SHARED CONTAINER VOLUME PATH ---
            SHARED_METRICS_PATH = "/app/shared/metrics.json"
            
            # Runtime safety: Ensure the /app/shared directory explicitly exists before writing
            os.makedirs(os.path.dirname(SHARED_METRICS_PATH), exist_ok=True)
            
            with open(SHARED_METRICS_PATH, "w") as f:
                json.dump(summary_metrics, f, indent=4)
            
            # Save detailed CSV reports safely
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv("data/processed/evaluation_report.csv", index=False)
            
            print("\n✅ Evaluation Complete!")
            print(f"Mean Faithfulness: {summary_metrics['faithfulness']:.2%}")
            print(f"Mean Relevancy: {summary_metrics['relevancy']:.2%}")
            print(f"Results exported directly to {SHARED_METRICS_PATH}")

            return summary_metrics
        
        except Exception as e:
            print(f"💥 Inner Evaluation Pipeline Failed: {e}")
            raise e

if __name__ == "__main__":
    run_evaluation()