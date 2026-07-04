import csv
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone.errors.exceptions import RateLimitError
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

    entries = []
    for chunk_index, chunk in enumerate(raw_chunks):
        # Each chunk stores section content + file-level metadata:
        #   "title"         — section heading   e.g. "5.0 Requirements"
        #   "text_content"  — body text         e.g. "All devices must..."
        #   "heading_level" — heading depth     e.g. 1 / 2 / 3
        #   "page"          — PDF page number where the chunk starts e.g. 21
        #   "source"        — filename          e.g. "D72001_REV03.pdf"
        #   "document_id"   — doc ID            e.g. "D72001"
        #   "revision"      — revision number   e.g. 3
        #   "folder_type"   — top-level folder  e.g. "design changes"
        #   "subfolder"     — subfolder (only if it exists) e.g. "concept proposal"
        #   "chunk_index"   — position of this chunk within its source file (0-based),
        #                     used to look up the preceding/following chunk of the same file
        entry = {
            "title":        chunk["title"],
            "text_content": chunk["text_content"],
            "heading_level": chunk["heading_level"],
            "page":         chunk["page"],
            "source":       meta["filename"],
            "document_id":  meta["document_id"],
            "revision":     meta["revision"],
            "folder_type":  meta["folder_type"],
            "chunk_index":  chunk_index,
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

    def retrieve(self, query: str, top_k: int = 5, max_per_source: int = 2) -> list[dict]:
        """
        Parameters
        ----------
        query          : 감사 질문 문자열
        top_k          : 반환할 상위 chunk 수
        max_per_source : 한 문서(source)당 결과에 포함할 최대 chunk 수 -- 길게 잘린 문서 하나가
                         (동일 heading 아래 "(Continued)" 조각들로) top_k를 독점하는 걸 방지

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
        candidate_pool = max(top_k * 10, 100)
        response = self.index.query(
            vector=query_vec,
            top_k=candidate_pool,
            include_metadata=True,
            # Cover pages / signature blocks / revision tables aren't real section
            # content, and their boilerplate legal/regulatory language embeds
            # deceptively close to audit questions -- exclude them at query time
            # rather than re-embedding the whole index.
            filter={"title": {"$nin": ["Document_Header", "Document_Header (Continued)"]}},
        )

        if not response.matches:
            print("[RetrievalAgent] No indexed documents available.")
            return []

        results = []
        per_source_count = {}
        for match in response.matches:
            md = match.metadata or {}
            source = md.get("source")
            if per_source_count.get(source, 0) >= max_per_source:
                continue
            per_source_count[source] = per_source_count.get(source, 0) + 1

            results.append({
                "rank": len(results) + 1,
                "score": float(match.score),
                "source": source,
                "document_id": md.get("document_id"),
                "revision": md.get("revision"),
                "chunk_index": md.get("chunk_index"),
                "title": md.get("title"),
                "heading_level": md.get("heading_level"),
                "page": md.get("page"),
                "text_content": md.get("text_content"),
            })
            if len(results) == top_k:
                break

        return results

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

        fetched = self.index.fetch(ids=ids).vectors

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
            }

        return {
            "before": [r for i in before_indices if (r := _to_result(i)) is not None],
            "after": [r for i in after_indices if (r := _to_result(i)) is not None],
        }
