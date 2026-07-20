"""
Ad-hoc test: retrieves the top 150 raw candidates (embedding similarity only,
no reranking) for a single fixed sub-query, so retrieval quality can be
inspected directly without running the full sub-query-generation + rerank +
synthesis pipeline.

Usage: python test_subquery_retrieval.py
"""
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import LATEST_REVISIONS_CSV, OUTPUT_DIR
from retrieval_agent import RetrievalAgent
from utils import CHUNK_CSV_FIELDNAMES as FIELDNAMES, write_dicts_to_csv

SUB_QUERY = "software identification, version control, and configuration management"
TOP_N = 150

TEST_RUNS_DIR = os.path.join(OUTPUT_DIR, "subquery_tests")


def main():
    agent = RetrievalAgent(docs_folder="docs", csv_path=LATEST_REVISIONS_CSV)

    # candidate_pool inside retrieve() is max(top_k * 30, 100) -- top_k=10 gives
    # a pool of 300, comfortably above TOP_N after diversification.
    results = agent.retrieve(SUB_QUERY, top_k=10, result_limit=TOP_N)
    print(f"  retrieved {len(results)} raw candidates for: {SUB_QUERY!r}")

    rows = []
    for r in results:
        row = dict(r)
        row["sub_query"] = SUB_QUERY
        row["rerank_score"] = ""
        rows.append(row)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(TEST_RUNS_DIR, f"{tag}_subquery_top150_raw.csv")
    write_dicts_to_csv(out_path, rows, FIELDNAMES)
    print(f"  -> saved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
