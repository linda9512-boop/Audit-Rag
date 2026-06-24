import os
import numpy as np
from sentence_transformers import SentenceTransformer
from pdf_chunking import parse_pdf_to_section_chunks


class RetrievalAgent:
    """
    Retrieval Agent: 감사 질문을 받아 관련 문서 chunk를 반환.

    Flow:
        1. docs_folder 안의 모든 PDF를 로드 & 청킹
        2. 각 chunk를 embedding으로 벡터화 (캐시)
        3. 쿼리 embedding과 코사인 유사도 계산
        4. 상위 top_k chunk 반환
    """

    def __init__(self, docs_folder: str, model_name: str = "all-MiniLM-L6-v2"):
        self.docs_folder = docs_folder
        self.model = SentenceTransformer(model_name)
        self.chunks = []          # list of dict: {title, text_content, source, heading_level}
        self.embeddings = None    # np.ndarray shape (N, D)

        print(f"[RetrievalAgent] Loading documents from: {docs_folder}")
        self._load_and_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_index(self):
        """docs_folder 내 PDF 파일을 모두 청킹 후 임베딩 인덱스 구축."""
        pdf_files = [
            f for f in os.listdir(self.docs_folder) if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            print("[RetrievalAgent] WARNING: No PDF files found in the folder.")
            return

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.docs_folder, pdf_file)
            print(f"  -> Chunking: {pdf_file}")
            try:
                raw_chunks = parse_pdf_to_section_chunks(pdf_path)
                for chunk in raw_chunks:
                    self.chunks.append({
                        "title": chunk["title"],
                        "text_content": chunk["text_content"],
                        "heading_level": chunk["heading_level"],
                        "source": pdf_file,
                    })
            except Exception as e:
                print(f"  [!] Failed to parse {pdf_file}: {e}")

        if not self.chunks:
            print("[RetrievalAgent] WARNING: No chunks were produced.")
            return

        print(f"[RetrievalAgent] Total chunks indexed: {len(self.chunks)}")
        print("[RetrievalAgent] Building embedding index...")

        texts = [
            f"{c['title']} {c['text_content']}" for c in self.chunks
        ]
        self.embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalised → dot product == cosine similarity
        )
        print("[RetrievalAgent] Index ready.\n")

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
