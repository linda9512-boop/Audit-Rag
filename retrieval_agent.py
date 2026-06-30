import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_chunking import parse_pdf_to_section_chunks
from extracting_latest import get_latest_revisions

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".vector_cache")
CACHE_EMBEDDINGS = os.path.join(CACHE_DIR, "embeddings.npy")
CACHE_CHUNKS     = os.path.join(CACHE_DIR, "chunks.json")


class RetrievalAgent:
    """
    Retrieval Agent: receives an audit question and returns relevant document chunks.

    Flow:
        1. Load & chunk all PDFs in docs_folder
        2. Embed each chunk (saved to .vector_cache/ so recomputation is skipped on next run)
        3. Compute cosine similarity against the query embedding
        4. Return top_k chunks
    """

    def __init__(self, docs_folder: str, model_name: str = "all-MiniLM-L6-v2"):
        self.docs_folder = docs_folder
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

        print(f"[RetrievalAgent] Loading documents from: {docs_folder}")
        self._load_and_index()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_exists(self) -> bool:
        return os.path.isfile(CACHE_EMBEDDINGS) and os.path.isfile(CACHE_CHUNKS)

    def _save_cache(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(CACHE_EMBEDDINGS, self.embeddings)
        with open(CACHE_CHUNKS, "w") as f:
            json.dump(self.chunks, f)
        print(f"[RetrievalAgent] Cache saved to {CACHE_DIR}")

    def _load_cache(self):
        self.embeddings = np.load(CACHE_EMBEDDINGS)
        with open(CACHE_CHUNKS) as f:
            self.chunks = json.load(f)
        print(f"[RetrievalAgent] Loaded {len(self.chunks)} chunks from cache.")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _load_and_index(self):
        """Chunk all PDFs and build the embedding index (or load from cache)."""
        if self._cache_exists():
            print("[RetrievalAgent] Cache found — skipping re-embedding.")
            self._load_cache()
            return

        pdf_metas = get_latest_revisions(self.docs_folder)

        if not pdf_metas:
            print("[RetrievalAgent] WARNING: No PDF files found in the folder.")
            return

        for meta in pdf_metas:
            print(f"  -> Chunking (latest revision): {meta['filename']}")
            try:
                raw_chunks = parse_pdf_to_section_chunks(meta["local_path"])
                for chunk in raw_chunks:
                    # Each chunk stores section content + file-level metadata:
                    #   "title"         — section heading   e.g. "5.0 Requirements"
                    #   "text_content"  — body text         e.g. "All devices must..."
                    #   "heading_level" — heading depth     e.g. 1 / 2 / 3
                    #   "source"        — filename          e.g. "D72001_REV03.pdf"
                    #   "document_id"   — doc ID            e.g. "D72001"
                    #   "revision"      — revision number   e.g. 3
                    #   "folder_type"   — top-level folder  e.g. "design changes"
                    #   "subfolder"     — subfolder (only if it exists) e.g. "concept proposal"
                    entry = {
                        "title":        chunk["title"],
                        "text_content": chunk["text_content"],
                        "heading_level": chunk["heading_level"],
                        "source":       meta["filename"],
                        "document_id":  meta["document_id"],
                        "revision":     meta["revision"],
                        "folder_type":  meta["folder_type"],
                    }
                    if "subfolder" in meta:
                        entry["subfolder"] = meta["subfolder"]
                    self.chunks.append(entry)
            except Exception as e:
                print(f"  [!] Failed to parse {meta['filename']}: {e}")

        if not self.chunks:
            print("[RetrievalAgent] WARNING: No chunks were produced.")
            return

        print(f"[RetrievalAgent] Total chunks: {len(self.chunks)} — building embeddings...")
        texts = [f"{c['title']} {c['text_content']}" for c in self.chunks]
        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalised → dot product == cosine similarity
        )
        self._save_cache()
        print("[RetrievalAgent] Index ready.\n")

    def clear_cache(self):
        """Delete cached vectors so the index is rebuilt on next run."""
        for path in (CACHE_EMBEDDINGS, CACHE_CHUNKS):
            if os.path.isfile(path):
                os.remove(path)
        print("[RetrievalAgent] Cache cleared.")

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
        """L2-normalised 벡터 기준 코사인 유사도 (dot product)."""
        return corpus_vecs @ query_vec  # shape (N,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Parameters
        ----------
        query   : 감사 질문 문자열
        top_k   : 반환할 상위 chunk 수

        Returns
        -------
        list of dict with keys:
            rank, score, source, title, heading_level, text_content
        """
        if self.embeddings is None or len(self.chunks) == 0:
            print("[RetrievalAgent] No indexed documents available.")
            return []

        query_vec = self.model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
        scores = self._cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append({
                "rank": rank,
                "score": float(scores[idx]),
                "source": self.chunks[idx]["source"],
                "title": self.chunks[idx]["title"],
                "heading_level": self.chunks[idx]["heading_level"],
                "text_content": self.chunks[idx]["text_content"],
            })

        return results
