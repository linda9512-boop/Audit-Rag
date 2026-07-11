import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieval_agent import RetrievalAgent
from synthesis_agent import synthesize

AUGMENT_MAX_WORKERS = 10  # each matched chunk's get_section_context()/get_document_header()
                          # is an independent chain of Pinecone fetch() calls -- run them
                          # concurrently instead of one chunk's whole chain at a time
MAX_TOTAL_CHUNKS = 200  # hard backstop on augment_chunks()'s output size (~70k tokens at
                        # ~350 tokens/chunk, safely under gpt-4o's 128k limit). Needed even
                        # with MAX_CONTINUATION_STEPS capped per-run, because several of the
                        # 30 matched chunks can independently land inside the SAME huge
                        # document (e.g. a 100+ chunk regulatory checklist) at different
                        # positions -- each pull is capped, but many independent pulls from
                        # one document can still add up past the window.

# The 27 audit sub-questions, in a fixed order used to number output files 01..27.
SUB_QUERIES = [
    "How does the organization determine which products/devices are subject to design and development controls?",
    "What criteria are used to classify a product as a medical device under applicable regulatory jurisdictions?",
    "Are all devices, including accessories, software, variants, and families, appropriately identified and included?",
    "How are changes such as maintenance releases, configuration changes, or software updates evaluated to determine if design controls apply?",
    "Are legacy devices assessed for compliance with current design control requirements?",
    "How does the organization ensure global regulatory requirements are considered in determining design control applicability?",
    "Does the organization have documented design and development procedures compliant with applicable regulations?",
    "Are these procedures consistently applied to all identified devices?",
    "How does the organization ensure design controls are applied across the lifecycle for new, modified, and legacy devices?",
    "Are roles and responsibilities for design activities clearly defined?",
    "How is risk integrated into design and development activities?",
    "Is technical documentation established and maintained for each device?",
    "Does the documentation meet regulatory requirements for applicable markets?",
    "Is there a defined structure for technical documentation, such as DHF, STED, or Technical File?",
    "How is completeness and consistency of technical documentation ensured?",
    "How are updates and revisions to technical documentation controlled?",
    "Is traceability established from requirements to design to verification and validation?",
    "Does the technical documentation align with regulatory requirements of all applicable jurisdictions?",
    "Are country-specific requirements addressed, such as FDA DHF, EU Technical File, or Health Canada STED?",
    "How does the organization ensure updates to regulatory requirements are reflected in technical documentation?",
    "How is technical documentation controlled, including versioning, approvals, and access?",
    "Who has responsibility for maintaining technical documentation?",
    "Is documentation readily retrievable for audit and regulatory submission?",
    "Are document control processes followed?",
    "Is each device uniquely identified and linked to its technical documentation?",
    "How are product variants and configurations managed within technical documentation?",
    "Is traceability maintained across product families and versions?",
]

CANDIDATE_POOL_PER_QUERY = 50  # raw candidates fetched by embedding search, before reranking
TOP_N_PER_QUERY = 30  # kept after reranking -- lets the cross-encoder actually filter, not just reorder
OUTPUT_DIR = "synthesis_answers"
CHUNKS_CSV = "retrieved_documents_per_query.csv"  # all rows, no cross-question dedup

FIELDNAMES = ["sub_query", "rerank_score", "score", "source", "document_id",
              "revision", "chunk_index", "title", "page", "text_content"]


def _key(c: dict) -> tuple:
    return (c.get("document_id"), c.get("revision"), c.get("chunk_index"))


def augment_chunks(agent: RetrievalAgent, chunks: list[dict]) -> list[dict]:
    """
    Add two kinds of extra context around a reranked top-N, without changing
    which chunks were judged most relevant:
      - section context for every chunk: whatever chunk(s) sit on the page before
        and the page after its own (see get_section_context() for why this is
        page-based, not a fixed +/-1 chunk_index step)
      - each unique document's own chunk_index 0 (title page, approval/revision
        history, scope -- usually "Document_Header"), so the model knows what the
        document actually is (e.g. initial release vs. a later revision) even when
        the matched chunk is from deep inside it
    """
    augmented = []
    seen = set()

    for c in chunks:
        c = dict(c)
        c["context_role"] = "matched"
        augmented.append(c)
        seen.add(_key(c))

    # Headers first (bounded by unique document count, typically well under 30,
    # and important for citation/identity accuracy), so they're never squeezed
    # out by the neighbor budget below.
    unique_docs = []
    seen_docs = set()
    for c in chunks:
        doc_key = (c.get("document_id"), c.get("revision"))
        if doc_key not in seen_docs:
            seen_docs.add(doc_key)
            unique_docs.append(c)

    with ThreadPoolExecutor(max_workers=AUGMENT_MAX_WORKERS) as executor:
        headers = list(executor.map(agent.get_document_header, unique_docs))

    for header in headers:
        if header is not None and _key(header) not in seen:
            seen.add(_key(header))
            header = dict(header)
            header["context_role"] = "header"
            augmented.append(header)

    # Each chunk's get_section_context() is its own independent chain of sequential
    # Pinecone fetch() calls -- independent across chunks, so run the 30 chains
    # concurrently rather than one full chain at a time.
    with ThreadPoolExecutor(max_workers=AUGMENT_MAX_WORKERS) as executor:
        context_lists = list(executor.map(agent.get_section_context, chunks))

    # Neighbors get whatever budget remains, filled in rerank-priority order (chunks
    # is already sorted best-first, and executor.map preserves input order), so if
    # the hard cap forces a cutoff, it's the lowest-ranked matched chunks' context
    # that gets dropped first, not the highest-ranked ones'.
    neighbor_budget = MAX_TOTAL_CHUNKS - len(augmented)
    for context in context_lists:
        if neighbor_budget <= 0:
            break
        for n in context:
            if neighbor_budget <= 0:
                break
            if _key(n) not in seen:
                seen.add(_key(n))
                n = dict(n)
                n["context_role"] = "neighbor"
                augmented.append(n)
                neighbor_budget -= 1

    return augmented


if __name__ == "__main__":
    agent = RetrievalAgent(docs_folder="docs", csv_path="latest_revisions.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    for i, question in enumerate(SUB_QUERIES, start=1):
        candidates = agent.retrieve(question, top_k=10, result_limit=CANDIDATE_POOL_PER_QUERY)
        chunks = agent.rerank(question, candidates, top_n=TOP_N_PER_QUERY)
        chunks = augment_chunks(agent, chunks)

        print(f"[{i:02d}/{len(SUB_QUERIES)}] ({len(chunks)} chunks) {question}")

        for c in chunks:
            row = dict(c)
            row["sub_query"] = question
            all_rows.append(row)

        answer = synthesize(question, chunks) if chunks else "[No retrieved chunks available for this question.]"

        out_path = os.path.join(OUTPUT_DIR, f"synthesis_answer_{i:02d}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"Audit question: {question}\n")
            f.write(f"Chunks used: {len(chunks)}\n")
            f.write("=" * 80 + "\n")
            f.write(answer + "\n")

        print(f"  -> saved to {out_path}\n")

    with open(CHUNKS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k) for k in FIELDNAMES})

    print(f"Done. {len(SUB_QUERIES)} answers saved to {OUTPUT_DIR}/, "
          f"{len(all_rows)} chunk rows saved to {CHUNKS_CSV}")
