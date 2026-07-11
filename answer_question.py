"""
Answers one audit question by splitting it into facet-specific sub-queries,
retrieving+reranking per facet, then synthesizing one final answer.

Usage: python answer_question.py "<audit question>"
Writes <tag>_subquery_raw_candidates.csv, <tag>_subquery_selected_top15.csv, and
<tag>_final_synthesis_answer.txt, where <tag> is a run timestamp (each output
file also has the question text written inside it).
"""
import csv
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieval_agent import RetrievalAgent
from synthesis_agent import synthesize
from synthesize_per_query import augment_chunks
from subquery_agent import generate_subqueries

CANDIDATE_POOL = 100  # raw candidates fetched by embedding search, before reranking
TOP_N_PER_SUBQUERY = 30  # kept after reranking, per sub-query (fixed, not a shared total)

FIELDNAMES = ["sub_query", "rerank_score", "score", "source", "document_id",
              "revision", "chunk_index", "title", "page", "text_content"]


def _key(c: dict) -> tuple:
    return (c.get("document_id"), c.get("revision"), c.get("chunk_index"))


def _parse_documents_cited(answer: str) -> list[str]:
    """Pull out the "Documents cited:" list at the end of a synthesize() answer
    (see rule 10 of SYSTEM_PROMPT in synthesis_agent.py) as a plain list of strings."""
    marker = "Documents cited:"
    idx = answer.find(marker)
    if idx == -1:
        return []
    tail = answer[idx + len(marker):]
    return [line.strip().lstrip("-").strip() for line in tail.splitlines() if line.strip()]


def run_question(agent: RetrievalAgent, question: str) -> dict:
    """Returns {"answer", "subqueries", "documents_cited", "chunks_used"} for a
    single arbitrary audit question."""
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"[{tag}] {question}")
    subqueries = generate_subqueries(question)
    per_subquery_n = TOP_N_PER_SUBQUERY
    print(f"  {len(subqueries)} sub-queries -> {per_subquery_n} chunks each")
    for sq in subqueries:
        print(f"  sub-query: {sq}")

    all_raw_rows = []
    all_selected_rows = []
    selected_chunks = []

    for sq in subqueries:
        candidates = agent.retrieve(sq, top_k=10, result_limit=CANDIDATE_POOL)
        reranked = agent.rerank(sq, candidates, top_n=per_subquery_n)

        print(f"  [{sq}] raw={len(candidates)} selected={len(reranked)}")
        for i, r in enumerate(reranked[:5], start=1):
            print(f"    {i}  {r['rerank_score']:.4f}  {r['document_id']}  {r['title']}")

        for c in candidates:
            row = dict(c)
            row["sub_query"] = sq
            row["rerank_score"] = ""
            all_raw_rows.append(row)

        for c in reranked:
            row = dict(c)
            row["sub_query"] = sq
            all_selected_rows.append(row)
            selected_chunks.append(c)

    raw_csv = f"{tag}_subquery_raw_candidates.csv"
    selected_csv = f"{tag}_subquery_selected_top15.csv"

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_raw_rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})

    with open(selected_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_selected_rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})

    print(f"  -> saved {len(all_raw_rows)} raw rows to {raw_csv}, "
          f"{len(all_selected_rows)} selected rows to {selected_csv}")

    deduped = {}
    for c in selected_chunks:
        key = _key(c)
        if key not in deduped or c["rerank_score"] > deduped[key]["rerank_score"]:
            deduped[key] = c
    matched_chunks = list(deduped.values())

    final_chunks = augment_chunks(agent, matched_chunks)
    print(f"  {len(matched_chunks)} deduped matched chunks -> "
          f"{len(final_chunks)} after page-based augmentation")

    answer = synthesize(question, final_chunks)

    out_path = f"{tag}_final_synthesis_answer.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Audit question: {question}\n")
        f.write(f"Sub-queries used: {subqueries}\n")
        f.write(f"Chunks used: {len(final_chunks)} "
                f"({len(matched_chunks)} matched from {len(subqueries)} facet sub-queries + augmented context)\n")
        f.write("=" * 80 + "\n")
        f.write(answer + "\n")

    print(f"  -> saved final answer to {out_path}\n")
    return {
        "answer": answer,
        "subqueries": subqueries,
        "documents_cited": _parse_documents_cited(answer),
        "chunks_used": len(final_chunks),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python answer_question.py "<audit question>"')
        sys.exit(1)

    agent = RetrievalAgent(docs_folder="docs", csv_path="latest_revisions.csv")

    question = sys.argv[1]
    result = run_question(agent, question)
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
