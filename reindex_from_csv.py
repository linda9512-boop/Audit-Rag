import os
import time

from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.errors.exceptions import NotFoundError

from retrieval_agent import RetrievalAgent, PINECONE_INDEX_NAME

load_dotenv()

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
CSV_PATH = os.path.join(os.path.dirname(__file__), "latest_revisions.csv")

if __name__ == "__main__":
    api_key = os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=api_key)
    index = pc.Index(PINECONE_INDEX_NAME)

    print(f"[1/2] Clearing all vectors from index '{PINECONE_INDEX_NAME}'...")
    try:
        index.delete(delete_all=True)
    except NotFoundError:
        print("  (namespace already empty)")

    for attempt in range(24):  # poll instead of a fixed sleep -- delete is eventually consistent
        time.sleep(5)
        remaining = index.describe_index_stats().total_vector_count
        if remaining == 0:
            break
        print(f"  ...still {remaining} vectors present, waiting (attempt {attempt + 1}/24)")
    else:
        raise RuntimeError("Index did not report 0 vectors after clearing -- aborting re-embed.")

    print(f"[2/2] Re-embedding files listed in {CSV_PATH}...")
    RetrievalAgent(docs_folder=DOCS_FOLDER, csv_path=CSV_PATH)
