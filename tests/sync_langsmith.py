# tests/sync_langsmith.py
from langsmith import Client
from tests.custom_eval import test_set 
from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
# Ensure this matches exactly what is in your browser URL for 'Workspace 1'
os.environ["LANGSMITH_WORKSPACE_ID"] = "YOUR_WORKSPACE_ID"

load_dotenv()

client = Client(
    api_key=os.environ.get("LANGCHAIN_API_KEY"), 
    # Use the standard endpoint unless you are on a private AWS instance
    api_url="https://aws.api.smith.langchain.com" 
    )

dataset_name = "RAG_Research_Papers"

if not client.has_dataset(dataset_name=dataset_name):
    print(f"Creating dataset: {dataset_name}")
    client.create_dataset(dataset_name=dataset_name)
        
    for item in test_set:
        client.create_example(
            inputs={"question": item["question"]},
            outputs={"ground_truth": item["ground_truth"]},
            dataset_name=dataset_name
        )
        print("✅ Sync complete!")
    else:
        print(f"Dataset '{dataset_name}' already exists in LangSmith.")
