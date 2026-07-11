import csv
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.errors.exceptions import ApiError, RateLimitError
from pdf_chunking import parse_pdf_to_section_chunks
from extracting_latest import get_latest_revisions

load_dotenv()


def _chunk_file(meta: dict) -> tuple[list[dict], str | None]:
    """Parse one PDF into chunk-entry dicts. Module-level (not a method) so it can
    be pickled and run in a worker process via ProcessPoolExecutor."""
    try:
        raw_chunks = parse_pdf_to_section_chunks(meta["local_path"])
    except Exception as e:
        return [], f"{meta['filename']}: {e}"

    # `chunk` already has title/text_content/heading_level/page from pdf_chunking.py --
    # spread those through as-is and add the file-level metadata pdf_chunking.py has
    # no way to know (source, document_id, revision, folder_type, chunk_index).
    entries = []
    for chunk_index, chunk in enumerate(raw_chunks):
        entry = {
            **chunk,
            "source":       meta["filename"],
            "document_id":  meta["document_id"],
            "revision":     meta["revision"],
            "folder_type":  meta["folder_type"],
            "chunk_index":  chunk_index,  # position within this file, used to look up neighbors
        }
        if "subfolder" in meta:
            entry["subfolder"] = meta["subfolder"]
        entries.append(entry)
    return entries, None


PINECONE_INDEX_NAME = "audit"
EMBED_MODEL = "llama-text-embed-v2"  # Pinecone-hosted model backing the "audit" index
EMBED_BATCH_SIZE = 96  # Pinecone inference API batch limit for text embedding models
EMBED_RATE_LIMIT_RETRIES = 6  # retries on 429 before giving up on a batch
EMBED_MAX_WORKERS = 10  # concurrent embed+upsert batches -- _embed_batch's own retry/backoff absorbs 429s
DOCUMENT_HEADER_PENALTY = 0.7  # score multiplier for cover-page/boilerplate chunks in retrieve()
RERANK_MODEL = "bge-reranker-v2-m3"  # Pinecone-hosted cross-encoder reranker
PINECONE_RETRIES = 2  # attempts on transient Pinecone API errors (e.g. network/proxy
                      # redirects) before giving up on a query/fetch call
PINECONE_RETRY_BASE_WAIT = 2  # seconds; doubles each retry (exponential backoff)


class RetrievalAgent:
    """
    Retrieval Agent: receives an audit question and returns relevant document chunks.

    Flow:
        1. Load & chunk all PDFs in docs_folder (skipped if the Pinecone index is already populated)
        2. Embed each chunk (via Pinecone's hosted llama-text-embed-v2) and upsert it into the "audit" index
        3. Query Pinecone for the closest chunks to the question embedding
        4. Return top_k chunks
    """

    def __init__(self, docs_folder: str, csv_path: str | None = None):
        self.docs_folder = docs_folder
        self.csv_path = csv_path

        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY not set. Add it to your .env file."
            )

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)

        print(f"[RetrievalAgent] Loading documents from: {docs_folder}")
        self._load_and_index()

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Embed a single batch (<= EMBED_BATCH_SIZE texts), retrying with backoff on rate limits."""
        for attempt in range(1, EMBED_RATE_LIMIT_RETRIES + 1):
            try:
                result = self.pc.inference.embed(
                    model=EMBED_MODEL,
                    inputs=texts,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                return [e.values for e in result.data]
            except RateLimitError:
                if attempt == EMBED_RATE_LIMIT_RETRIES:
                    raise
                wait_s = 60
                print(f"  [rate limit] waiting {wait_s}s (attempt {attempt}/{EMBED_RATE_LIMIT_RETRIES})...")
                time.sleep(wait_s)

    # ------------------------------------------------------------------
    # Pinecone helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_id(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", text)

    @classmethod
    def _file_key(cls, document_id, revision, source: str) -> str:
        """Stable per-file key used to build vector IDs, so a chunk's neighbors
        (chunk_index - 1 / + 1 of the same file) can be fetched by ID directly."""
        doc_id = document_id or cls._sanitize_id(Path(source).stem)
        rev_part = revision if revision is not None else 0
        return f"{doc_id}-r{rev_part}"

    def _upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        vectors = []
        for chunk, vec in zip(chunks, embeddings):
            file_key = self._file_key(chunk.get("document_id"), chunk.get("revision"), chunk["source"])
            vector_id = f"{file_key}-c{chunk['chunk_index']}"
            metadata = {k: v for k, v in chunk.items() if v is not None}
            vectors.append({"id": vector_id, "values": vec, "metadata": metadata})

        self.index.upsert(vectors=vectors)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _load_metas_from_csv(csv_path: str) -> list[dict]:
        """Read a latest_revisions.csv (as produced by extracting_latest.save_latest_revisions_csv)
        into the same meta dict shape get_latest_revisions() returns."""
        metas = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                revision = row.get("revision") or ""
                meta = {
                    "local_path":  row["local_path"],
                    "filename":    row["filename"],
                    "document_id": row.get("document_id") or None,
                    "revision":    int(revision) if revision.strip() else None,
                    "folder_type": row.get("folder_type") or None,
                }
                if row.get("subfolder"):
                    meta["subfolder"] = row["subfolder"]
                metas.append(meta)
        return metas

    def _load_and_index(self):
        """Chunk all PDFs and upsert embeddings into Pinecone (skipped if already populated)."""
        stats = self.index.describe_index_stats()
        if stats.total_vector_count > 0:
            print(f"[RetrievalAgent] Index already has {stats.total_vector_count} vectors — skipping re-embedding.")
            return

        if self.csv_path:
            print(f"[RetrievalAgent] Loading file list from: {self.csv_path}")
            pdf_metas = self._load_metas_from_csv(self.csv_path)
        else:
            pdf_metas = get_latest_revisions(self.docs_folder)

        if not pdf_metas:
            print("[RetrievalAgent] WARNING: No PDF files found in the folder.")
            return

        chunks = []
        print(f"[RetrievalAgent] Chunking {len(pdf_metas)} files in parallel "
              f"(pool size {os.cpu_count()})...")
        done = 0
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_chunk_file, meta): meta for meta in pdf_metas}
            for future in as_completed(futures):
                meta = futures[future]
                entries, error = future.result()
                if error:
                    print(f"  [!] Failed to parse {error}")
                else:
                    chunks.extend(entries)
                done += 1
                if done % 20 == 0 or done == len(pdf_metas):
                    print(f"  [chunk] {done}/{len(pdf_metas)} files done")

        if not chunks:
            print("[RetrievalAgent] WARNING: No chunks were produced.")
            return

        batches = [chunks[i:i + EMBED_BATCH_SIZE] for i in range(0, len(chunks), EMBED_BATCH_SIZE)]
        print(f"[RetrievalAgent] Total chunks: {len(chunks)} — embedding + upserting via '{EMBED_MODEL}' "
              f"in {len(batches)} batches of {EMBED_BATCH_SIZE} ({EMBED_MAX_WORKERS} workers)...")
        done = 0
        with ThreadPoolExecutor(max_workers=EMBED_MAX_WORKERS) as executor:
            futures = [executor.submit(self._embed_and_upsert_batch, b) for b in batches]
            for future in as_completed(futures):
                future.result()  # surface exceptions from the worker thread
                done += 1
                if done % 5 == 0 or done == len(batches):
                    print(f"  [index] {done}/{len(batches)} batches upserted")

        print("[RetrievalAgent] Index ready.\n")

    def _embed_and_upsert_batch(self, batch_chunks: list[dict]):
        texts = [f"{c['title']} {c['text_content']}" for c in batch_chunks]
        embeddings = self._embed_batch(texts, input_type="passage")
        self._upsert_chunks(batch_chunks, embeddings)

    def clear_index(self):
        """Delete all vectors from the Pinecone index so it is rebuilt on next run."""
        self.index.delete(delete_all=True)
        print(f"[RetrievalAgent] Cleared all vectors from index '{PINECONE_INDEX_NAME}'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5, max_per_source: int | None = None, result_limit: int = 30) -> list[dict]:
        """
        Parameters
        ----------
        query          : the audit question string
        top_k          : number of top chunks to return
        max_per_source : max chunks to include per document (source) -- prevents one
                         long, heavily-split document (many "(Continued)" pieces under
                         the same heading) from monopolizing top_k.
                         None (default) disables this cap entirely -- confirmed via direct
                         score inspection that a long document's lower-ranked continuation
                         chunks were losing out to genuinely lower embedding similarity, not
                         to this cap, so capping was only ever costing coverage for no benefit.
        result_limit   : max number of diversified chunks to return (was a hardcoded 30)

        Returns
        -------
        list of dict with keys:
            rank, score, source, document_id, revision, chunk_index, title,
            heading_level, page, text_content
        """
        query_vec = self._embed_batch([query], input_type="query")[0]
        # Pull a much larger candidate pool than top_k -- diversification below discards
        # anything past max_per_source for a given document, so enough headroom is needed
        # for other documents to surface.
        candidate_pool = max(top_k * 30, 100)
        for attempt in range(1, PINECONE_RETRIES + 1):
            try:
                response = self.index.query(
                    vector=query_vec,
                    top_k=candidate_pool,
                    include_metadata=True,
                )
                break
            except ApiError:
                if attempt == PINECONE_RETRIES:
                    raise
                wait_s = PINECONE_RETRY_BASE_WAIT * (2 ** (attempt - 1))
                print(f"  [query error] retrying in {wait_s}s "
                      f"(attempt {attempt}/{PINECONE_RETRIES})...")
                time.sleep(wait_s)

        if not response.matches:
            print("[RetrievalAgent] No indexed documents available.")
            return []

        # Cover pages / signature blocks / revision tables are usually boilerplate whose
        # legal/regulatory language embeds deceptively close to audit questions -- but
        # occasionally a heading genuinely wasn't detected (e.g. before the Appendix/Annex
        # fix) and real content ended up under this title. A soft penalty lets them still
        # surface if they're a strong enough match, instead of a hard filter that would
        # silently drop them no matter how relevant.
        def _adjusted_score(match) -> float:
            score = float(match.score)
            title = (match.metadata or {}).get("title", "")
            if title in ("Document_Header", "Document_Header (Continued)"):
                return score * DOCUMENT_HEADER_PENALTY
            return score

        ranked_matches = sorted(response.matches, key=_adjusted_score, reverse=True)

        results = []
        per_source_count = {}
        for match in ranked_matches:
            md = match.metadata or {}
            source = md.get("source")
            if max_per_source is not None and per_source_count.get(source, 0) >= max_per_source:
                continue
            per_source_count[source] = per_source_count.get(source, 0) + 1

            results.append({
                "rank": len(results) + 1,
                "score": _adjusted_score(match),
                "source": source,
                "document_id": md.get("document_id"),
                "revision": md.get("revision"),
                "chunk_index": md.get("chunk_index"),
                "title": md.get("title"),
                "heading_level": md.get("heading_level"),
                "page": md.get("page"),
                "text_content": md.get("text_content"),
                "folder_type": md.get("folder_type"),
                "subfolder": md.get("subfolder"),
            })
            if len(results) == result_limit:
                break

        return results

    def _fetch_ids(self, ids: list[str]) -> dict:
        """self.index.fetch(ids=...).vectors, wrapped with the same retry-on-ApiError
        behavior as retrieve()'s query call -- shared by get_neighbors(),
        get_section_context(), and get_document_header(), which all fetch chunks
        directly by ID rather than by similarity search."""
        for attempt in range(1, PINECONE_RETRIES + 1):
            try:
                return self.index.fetch(ids=ids).vectors
            except ApiError:
                if attempt == PINECONE_RETRIES:
                    raise
                wait_s = PINECONE_RETRY_BASE_WAIT * (2 ** (attempt - 1))
                print(f"  [fetch error] retrying in {wait_s}s "
                      f"(attempt {attempt}/{PINECONE_RETRIES})...")
                time.sleep(wait_s)

    def get_neighbors(self, chunk: dict, before: int = 1, after: int = 1) -> dict:
        """
        Fetch the chunks immediately preceding/following a chunk returned by retrieve(),
        within the same source file, by reconstructing their vector IDs and fetching
        them directly (no similarity search involved).

        Parameters
        ----------
        chunk  : a dict from retrieve()'s results (must include source, document_id,
                 revision, chunk_index)
        before : how many preceding chunks to fetch
        after  : how many following chunks to fetch

        Returns
        -------
        {"before": [...chunks in order...], "after": [...chunks in order...]}
        Missing neighbors (e.g. at the start/end of a file) are simply omitted.
        """
        file_key = self._file_key(chunk.get("document_id"), chunk.get("revision"), chunk["source"])
        idx = chunk["chunk_index"]

        before_indices = list(range(max(0, idx - before), idx))
        after_indices = list(range(idx + 1, idx + 1 + after))
        ids = [f"{file_key}-c{i}" for i in before_indices + after_indices]

        fetched = self._fetch_ids(ids)

        def _to_result(i: int) -> dict | None:
            vec = fetched.get(f"{file_key}-c{i}")
            if vec is None:
                return None
            md = vec.metadata or {}
            return {
                "source": md.get("source"),
                "document_id": md.get("document_id"),
                "revision": md.get("revision"),
                "chunk_index": md.get("chunk_index"),
                "title": md.get("title"),
                "heading_level": md.get("heading_level"),
                "page": md.get("page"),
                "text_content": md.get("text_content"),
                "folder_type": md.get("folder_type"),
                "subfolder": md.get("subfolder"),
            }

        return {
            "before": [r for i in before_indices if (r := _to_result(i)) is not None],
            "after": [r for i in after_indices if (r := _to_result(i)) is not None],
        }

    @staticmethod
    def _parse_page_range(page) -> tuple[int, int]:
        """"3" -> (3, 3); "3-5" -> (3, 5), matching format_page_range() in pdf_chunking.py."""
        s = str(page)
        if "-" in s:
            a, b = s.split("-", 1)
            return int(a), int(b)
        return int(s), int(s)

    def get_section_context(self, chunk: dict) -> list[dict]:
        """
        Return whatever chunk(s) sit on the page immediately before `chunk`'s own
        page range and immediately after it -- by page (a fixed boundary, so this
        doesn't cascade through the rest of the document), not a fixed chunk_index
        step, so a neighboring page holding several short chunks is captured in
        full. Applied uniformly to every chunk, "(Continued)" or not.

        Returns a list of chunks (not including `chunk` itself), in no particular
        order. Missing/out-of-range neighbors are simply omitted.
        """
        file_key = self._file_key(chunk.get("document_id"), chunk.get("revision"), chunk["source"])
        idx = chunk["chunk_index"]

        def fetch_one(i: int) -> dict | None:
            if i < 0:
                return None
            fetched = self._fetch_ids([f"{file_key}-c{i}"])
            vec = fetched.get(f"{file_key}-c{i}")
            if vec is None:
                return None
            md = vec.metadata or {}
            return {
                "source": md.get("source"),
                "document_id": md.get("document_id"),
                "revision": md.get("revision"),
                "chunk_index": md.get("chunk_index"),
                "title": md.get("title"),
                "heading_level": md.get("heading_level"),
                "page": md.get("page"),
                "text_content": md.get("text_content"),
                "folder_type": md.get("folder_type"),
                "subfolder": md.get("subfolder"),
            }

        def collect_by_page(start_idx: int, step: int, boundary_page: int) -> list[dict]:
            """Walk chunk_index by `step` (-1 or +1) from start_idx, collecting
            chunks as long as each one's page range still touches within 1 page of
            the FIXED boundary_page -- fixed, not expanding, so this only ever
            reaches the single neighboring page, however many short chunks share
            it, and never cascades further into the document."""
            results = []
            i = start_idx
            while True:
                c = fetch_one(i)
                if c is None:
                    break
                p_start, p_end = self._parse_page_range(c["page"])
                if step < 0:
                    if p_end < boundary_page - 1:
                        break
                else:
                    if p_start > boundary_page + 1:
                        break
                results.append(c)
                i += step
            return results

        own_start, own_end = self._parse_page_range(chunk["page"])
        before = collect_by_page(idx - 1, -1, own_start)
        after = collect_by_page(idx + 1, 1, own_end)

        return before + after

    def get_document_header(self, chunk: dict) -> dict | None:
        """
        Fetch chunk_index 0 (the "Document_Header" chunk -- everything before the
        first detected heading, e.g. title page, approval/revision history, scope)
        for the same file as `chunk`, by ID (no similarity search).

        Lets a document's own identity/purpose (e.g. "Initial release for Initial
        NMPA submission" vs. a later revision) travel alongside a retrieved chunk
        even when that chunk is from deep in the document, without relying on the
        embedding search to separately surface the header on its own merits.

        Returns None if `chunk` already is the header, or if it's missing (e.g.
        the file has no content before its first heading).
        """
        if chunk.get("chunk_index") == 0:
            return None

        file_key = self._file_key(chunk.get("document_id"), chunk.get("revision"), chunk["source"])
        fetched = self._fetch_ids([f"{file_key}-c0"])
        vec = fetched.get(f"{file_key}-c0")
        if vec is None:
            return None

        md = vec.metadata or {}
        return {
            "source": md.get("source"),
            "document_id": md.get("document_id"),
            "revision": md.get("revision"),
            "chunk_index": md.get("chunk_index"),
            "title": md.get("title"),
            "heading_level": md.get("heading_level"),
            "page": md.get("page"),
            "text_content": md.get("text_content"),
            "folder_type": md.get("folder_type"),
            "subfolder": md.get("subfolder"),
        }

    def rerank(self, query: str, chunks: list[dict], top_n: int | None = None) -> list[dict]:
        """
        Re-score chunks (typically the union of several retrieve() calls, e.g. from
        different sub-queries) against `query` using a single cross-encoder pass.

        This matters when combining results from multiple sub-queries: each sub-query's
        raw cosine scores come from a different query vector, so they aren't comparable
        to each other -- a chunk that's the best match for one sub-query can have a lower
        raw score than an unrelated chunk from a sub-query that happens to score high
        across the board. Reranking scores every candidate against the same query with
        the same model, producing one consistent ranking regardless of which sub-query
        originally surfaced each chunk.

        Parameters
        ----------
        query  : the query to rerank against (usually the original, non-decomposed question)
        chunks : chunk dicts (each must have "title" and "text_content"), e.g. the deduped
                 union of multiple retrieve() calls
        top_n  : how many top-ranked chunks to return (defaults to all of them, re-sorted)

        Returns
        -------
        chunks re-sorted by relevance, each with a "rerank_score" key added
        """
        if not chunks:
            return []

        documents = [{"text": f"{c['title']} {c['text_content']}"} for c in chunks]
        result = self.pc.inference.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents,
            rank_fields=["text"],
            top_n=top_n or len(chunks),
            return_documents=False,
        )

        reranked = []
        for ranked_doc in result.data:
            chunk = dict(chunks[ranked_doc.index])
            chunk["rerank_score"] = ranked_doc.score
            reranked.append(chunk)
        return reranked
