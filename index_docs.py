import os
from config import LATEST_REVISIONS_CSV
from retrieval_agent import RetrievalAgent

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
CSV_PATH = os.path.join(os.path.dirname(__file__), LATEST_REVISIONS_CSV)

if __name__ == "__main__":
    print(f"Chunking + embedding files listed in {CSV_PATH} into Pinecone index 'audit'...")
    RetrievalAgent(docs_folder=DOCS_FOLDER, csv_path=CSV_PATH)
