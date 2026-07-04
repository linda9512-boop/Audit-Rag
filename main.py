import os
from retrieval_agent import RetrievalAgent

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
TOP_K = 5


def print_results(results: list[dict]):
    if not results:
        print("\n[No relevant documents found.]\n")
        return

    print(f"\n{'='*60}")
    print(f"  TOP {len(results)} RELEVANT CHUNKS")
    print(f"{'='*60}")
    for r in results:
        print(f"\n[Rank {r['rank']}]  Score: {r['score']:.4f}")
        print(f"  Source : {r['source']}")
        print(f"  Section: {r['title']}  (Level {r['heading_level']}, Page {r['page']})")
        print(f"  Preview: {r['text_content'][:200]}{'...' if len(r['text_content']) > 200 else ''}")
    print(f"\n{'='*60}\n")


def main():
    # ------------------------------------------------------------------
    # 1. Setup: docs/ 폴더 확인
    # ------------------------------------------------------------------
    if not os.path.isdir(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"[Setup] Created docs folder at: {DOCS_FOLDER}")
        print("[Setup] Please place your PDF files in the 'docs/' folder and restart.\n")
        return

    # ------------------------------------------------------------------
    # 2. Retrieval Agent 초기화 (PDF 로드 + 인덱스 구축)
    # ------------------------------------------------------------------
    agent = RetrievalAgent(docs_folder=DOCS_FOLDER)

    # ------------------------------------------------------------------
    # 3. 감사 질문 입력 루프
    # ------------------------------------------------------------------
    print("Audit RAG System — Retrieval Agent")
    print("Type your audit question. Enter 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Audit Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Session ended]")
            break

        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("[Session ended]")
            break

        # ------------------------------------------------------------------
        # 4. 리트리벌 실행 & 결과 출력
        # ------------------------------------------------------------------
        results = agent.retrieve(query, top_k=TOP_K)
        print_results(results)


if __name__ == "__main__":
    main()
