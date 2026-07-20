"""
Answers one audit question by splitting it into facet-specific sub-queries,
retrieving+reranking per facet, then synthesizing one final answer.

Usage: python answer_question.py "<audit question>"
Writes <tag>_subquery_raw_candidates.csv, <tag>_subquery_selected_top15.csv, and
<tag>_final_synthesis_answer.txt into outputs/question_runs/, where <tag> is a
run timestamp (each output file also has the question text written inside it).
"""
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import ANSWER_CANDIDATE_POOL, ANSWER_TOP_N_PER_SUBQUERY, LATEST_REVISIONS_CSV, OUTPUT_DIR
from retrieval_agent import RetrievalAgent, augment_chunks
from synthesis_agent import synthesize, synthesize_stream
from subquery_agent import generate_subqueries
from timing_log import reset_timing, save_timing, tlog
from utils import CHUNK_CSV_FIELDNAMES as FIELDNAMES, chunk_key as _key, write_answer_file, write_dicts_to_csv

QUESTION_RUNS_DIR = os.path.join(OUTPUT_DIR, "question_runs")
TIMING_LOG_PATH = os.path.join(QUESTION_RUNS_DIR, "timing.log")


# Matches a bracketed OR parenthesized citation group, e.g. both
# "[D54978 ... Rev 5.0.pdf, page 5]" and "(D101277, p. 229; 104993953MIN-001, p. 235)"
# -- the model doesn't reliably stick to one bracket style or "page" spelling.
_CITATION_GROUP_RE = re.compile(r"[\[\(]([^\[\]\(\)]+)[\]\)]")
# Within a group, one or more "source, p[age][s]. N[-M]" items separated by ';'.
_CITATION_ITEM_RE = re.compile(r"^\s*(.+?),\s*p(?:age)?s?\.?\s*(\d+(?:[\-–]\d+)?)\s*$", re.IGNORECASE)


def _parse_documents_cited(answer: str) -> list[str]:
    """Extract every [source, page] citation directly from the answer body via
    regex, deduplicated in first-seen order. This doesn't rely on the model
    correctly compiling its own "Documents cited:" list at the end -- that
    formatting instruction (SYSTEM_PROMPT rule 10, now removed) turned out to
    be followed inconsistently by the time the model reaches the end of a long
    answer. It also doesn't assume a single inline citation style: the model
    has been observed switching between "[source, page N]" and
    "(source, p. N; other source, p. M)" even within one answer, so groups are
    matched with either bracket style and split on ';' for multi-citation
    groups before parsing each "source, page" pair."""
    seen = set()
    citations = []
    for group in _CITATION_GROUP_RE.findall(answer):
        for item in group.split(";"):
            m = _CITATION_ITEM_RE.match(item)
            if not m:
                continue
            source, page = m.group(1).strip(), m.group(2)
            key = (source, page)
            if key not in seen:
                seen.add(key)
                citations.append(f"{source}, page {page}")
    return citations


def _normalize_citations(answer: str) -> str:
    """Rewrite every recognized citation group -- whichever bracket style or
    "page" spelling the model actually used -- into the canonical
    "[source, page]" form, one bracket per citation, so the answer text shown
    to the user is visually consistent even when the model didn't produce that
    format itself. A group with any item that doesn't parse as a citation is
    left untouched (it's probably just a parenthetical aside, not a citation)."""
    def _replace(match: re.Match) -> str:
        items = match.group(1).split(";")
        parsed = [_CITATION_ITEM_RE.match(item) for item in items]
        if not all(parsed):
            return match.group(0)
        return " ".join(f"[{m.group(1).strip()}, page {m.group(2)}]" for m in parsed)

    return _CITATION_GROUP_RE.sub(_replace, answer)


def _retrieve_and_rerank(agent: RetrievalAgent, sq: str, per_subquery_n: int) -> dict:
    """One sub-query's retrieve+rerank, timed independently -- run concurrently
    across sub-queries by run_question() since they're fully independent of
    each other (each hits Pinecone with its own query vector)."""
    t_sq = time.perf_counter()
    candidates = agent.retrieve(sq, top_k=10, result_limit=ANSWER_CANDIDATE_POOL)
    t_retrieve = time.perf_counter()
    reranked = agent.rerank(sq, candidates, top_n=per_subquery_n)
    t_rerank = time.perf_counter()
    return {
        "sq": sq,
        "candidates": candidates,
        "reranked": reranked,
        "retrieve_time": t_retrieve - t_sq,
        "rerank_time": t_rerank - t_retrieve,
    }


def _prepare_context(agent: RetrievalAgent, question: str) -> dict:
    """Everything before synthesis: sub-query generation, retrieve+rerank per
    sub-query (concurrent), dedup, and page-based augmentation. Shared by
    run_question() (returns the full answer at once) and run_question_stream()
    (streams the answer as it's generated) -- they differ only in the final
    synthesis step."""
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    t_start = time.perf_counter()
    reset_timing()

    tlog(f"[{tag}] {question}")
    t0 = time.perf_counter()
    subqueries = generate_subqueries(question)
    tlog(f"  [timing] generate_subqueries: {time.perf_counter() - t0:.2f}s")
    per_subquery_n = ANSWER_TOP_N_PER_SUBQUERY
    print(f"  {len(subqueries)} sub-queries -> {per_subquery_n} chunks each")
    for sq in subqueries:
        print(f"  sub-query: {sq}")

    all_raw_rows = []
    all_selected_rows = []
    selected_chunks = []

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(subqueries)) as executor:
        sq_results = list(executor.map(
            lambda sq: _retrieve_and_rerank(agent, sq, per_subquery_n), subqueries
        ))
    tlog(f"  [timing] retrieve+rerank loop total ({len(subqueries)} sub-queries, concurrent): "
         f"{time.perf_counter() - t0:.2f}s")

    for res in sq_results:
        sq, candidates, reranked = res["sq"], res["candidates"], res["reranked"]
        tlog(f"  [{sq}] raw={len(candidates)} selected={len(reranked)} "
             f"[timing] retrieve={res['retrieve_time']:.2f}s rerank={res['rerank_time']:.2f}s")
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

    raw_csv = os.path.join(QUESTION_RUNS_DIR, f"{tag}_subquery_raw_candidates.csv")
    selected_csv = os.path.join(QUESTION_RUNS_DIR, f"{tag}_subquery_selected_top15.csv")

    write_dicts_to_csv(raw_csv, all_raw_rows, FIELDNAMES)
    write_dicts_to_csv(selected_csv, all_selected_rows, FIELDNAMES)

    print(f"  -> saved {len(all_raw_rows)} raw rows to {raw_csv}, "
          f"{len(all_selected_rows)} selected rows to {selected_csv}")

    deduped = {}
    for c in selected_chunks:
        key = _key(c)
        if key not in deduped or c["rerank_score"] > deduped[key]["rerank_score"]:
            deduped[key] = c
    matched_chunks = list(deduped.values())

    t0 = time.perf_counter()
    final_chunks = augment_chunks(agent, matched_chunks)
    tlog(f"  {len(matched_chunks)} deduped matched chunks -> "
         f"{len(final_chunks)} after page-based augmentation "
         f"[timing] augment_chunks: {time.perf_counter() - t0:.2f}s")

    return {
        "tag": tag,
        "t_start": t_start,
        "subqueries": subqueries,
        "matched_chunks": matched_chunks,
        "final_chunks": final_chunks,
    }


def _write_final_answer(tag: str, question: str, answer: str, subqueries: list[str],
                         matched_chunks: list[dict], final_chunks: list[dict]):
    out_path = os.path.join(QUESTION_RUNS_DIR, f"{tag}_final_synthesis_answer.txt")
    write_answer_file(
        out_path, question, answer,
        chunks_used=f"{len(final_chunks)} ({len(matched_chunks)} matched from "
                    f"{len(subqueries)} facet sub-queries + augmented context)",
        extra_lines=[f"Sub-queries used: {subqueries}"],
    )
    print(f"  -> saved final answer to {out_path}\n")


def run_question(agent: RetrievalAgent, question: str) -> dict:
    """Returns {"answer", "subqueries", "documents_cited", "chunks_used"} for a
    single arbitrary audit question."""
    ctx = _prepare_context(agent, question)
    tag, subqueries = ctx["tag"], ctx["subqueries"]
    matched_chunks, final_chunks = ctx["matched_chunks"], ctx["final_chunks"]

    t0 = time.perf_counter()
    answer = synthesize(question, final_chunks)
    tlog(f"  [timing] synthesize (LLM call): {time.perf_counter() - t0:.2f}s")
    tlog(f"  [timing] TOTAL run_question: {time.perf_counter() - ctx['t_start']:.2f}s")

    answer = _normalize_citations(answer)
    _write_final_answer(tag, question, answer, subqueries, matched_chunks, final_chunks)
    save_timing(TIMING_LOG_PATH)
    return {
        "answer": answer,
        "subqueries": subqueries,
        "documents_cited": _parse_documents_cited(answer),
        "chunks_used": len(final_chunks),
    }


def run_question_stream(agent: RetrievalAgent, question: str):
    """Like run_question(), but yields dicts as the answer streams in:
      {"type": "delta", "text": "..."}  -- one per streamed text chunk from the LLM
      {"type": "done", "subqueries": [...], "documents_cited": [...], "chunks_used": N}
                                         -- exactly once, after the full answer is in
    """
    ctx = _prepare_context(agent, question)
    tag, subqueries = ctx["tag"], ctx["subqueries"]
    matched_chunks, final_chunks = ctx["matched_chunks"], ctx["final_chunks"]

    t0 = time.perf_counter()
    answer_parts = []
    for delta in synthesize_stream(question, final_chunks):
        answer_parts.append(delta)
        yield {"type": "delta", "text": delta}
    answer = _normalize_citations("".join(answer_parts))
    tlog(f"  [timing] synthesize (LLM call, streamed): {time.perf_counter() - t0:.2f}s")
    tlog(f"  [timing] TOTAL run_question_stream: {time.perf_counter() - ctx['t_start']:.2f}s")

    _write_final_answer(tag, question, answer, subqueries, matched_chunks, final_chunks)
    save_timing(TIMING_LOG_PATH)
    yield {
        "type": "done",
        "subqueries": subqueries,
        "documents_cited": _parse_documents_cited(answer),
        "chunks_used": len(final_chunks),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python answer_question.py "<audit question>"')
        sys.exit(1)

    agent = RetrievalAgent(docs_folder="docs", csv_path=LATEST_REVISIONS_CSV)

    question = sys.argv[1]
    result = run_question(agent, question)
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
