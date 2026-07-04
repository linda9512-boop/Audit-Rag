"""
Audit RAG Pipeline
==================
Input (audit question)
  → Routing Agent   : Analyze question, extract search keywords
  → Retrieval Agent : Search relevant document chunks
  → Synthesis Agent : Generate answer draft from chunks
  → Critic Agent    : Verify answer completeness
  → Final Output    : Final audit response
"""

import os
import anthropic
from retrieval_agent import RetrievalAgent

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "docs")
TOP_K = 5
LLM_MODEL = "claude-opus-4-6"


# ------------------------------------------------------------------
# LLM helper
# ------------------------------------------------------------------
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


def call_llm(system: str, user: str) -> str:
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": user}],
        system=system,
    )
    return message.content[0].text.strip()


# ------------------------------------------------------------------
# Agent 1: Routing Agent
# ------------------------------------------------------------------
def routing_agent(query: str) -> dict:
    """
    Analyze the question and determine the retrieval strategy.
    Returns:
        {
          "refined_query": str,       # query optimized for document retrieval
          "keywords": list[str],      # key technical/regulatory keywords
          "doc_type_hint": str        # type of document evidence likely needed
        }
    """
    system = """You are an audit document routing assistant.
Given an audit question, extract:
1. A refined search query optimized for document retrieval
2. Key technical/regulatory keywords
3. What type of document evidence is likely needed

Respond in this exact format:
REFINED_QUERY: <query>
KEYWORDS: <keyword1>, <keyword2>, <keyword3>
DOC_TYPE: <type of document evidence needed>"""

    response = call_llm(system, f"Audit question: {query}")

    result = {"refined_query": query, "keywords": [], "doc_type_hint": "general"}
    for line in response.splitlines():
        if line.startswith("REFINED_QUERY:"):
            result["refined_query"] = line.replace("REFINED_QUERY:", "").strip()
        elif line.startswith("KEYWORDS:"):
            result["keywords"] = [k.strip() for k in line.replace("KEYWORDS:", "").split(",")]
        elif line.startswith("DOC_TYPE:"):
            result["doc_type_hint"] = line.replace("DOC_TYPE:", "").strip()

    return result


# ------------------------------------------------------------------
# Agent 3: Synthesis Agent
# ------------------------------------------------------------------
def synthesis_agent(query: str, chunks: list[dict]) -> str:
    """
    Read retrieved chunks and generate a draft answer to the audit question.
    """
    if not chunks:
        return "No relevant document evidence was found to answer this question."

    context_parts = []
    for c in chunks:
        context_parts.append(
            f"[Source: {c['source']} | Section: {c['title']} | Page: {c['page']}]\n{c['text_content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    system = """You are an audit response specialist.
Using only the provided document evidence, answer the audit question clearly and concisely.
- Cite which source document supports each claim.
- Do not add information beyond what is in the evidence.
- Be direct and factual."""

    user = f"""Audit Question: {query}

Document Evidence:
{context}

Generate a structured audit response based solely on the above evidence."""

    return call_llm(system, user)


# ------------------------------------------------------------------
# Agent 4: Critic Agent
# ------------------------------------------------------------------
def critic_agent(query: str, answer: str, chunks: list[dict]) -> dict:
    """
    Verify the completeness of the generated answer.
    Returns:
        {
          "verdict": "Fully Answered" | "Partially Answered" | "Not Answered",
          "missing_evidence": str,
          "confidence": float (0.0 ~ 1.0)
        }
    """
    sources = list({c["source"] for c in chunks})

    system = """You are an audit quality reviewer.
Evaluate whether the provided answer sufficiently addresses the audit question given the evidence.

Respond in this exact format:
VERDICT: <Fully Answered | Partially Answered | Not Answered>
CONFIDENCE: <0.0 to 1.0>
MISSING_EVIDENCE: <what additional evidence would be needed, or 'None'>"""

    user = f"""Audit Question: {query}

Generated Answer:
{answer}

Source Documents Used: {', '.join(sources) if sources else 'None'}

Evaluate the answer's completeness."""

    response = call_llm(system, user)

    result = {
        "verdict": "Partially Answered",
        "confidence": 0.5,
        "missing_evidence": "Unknown",
    }
    for line in response.splitlines():
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("MISSING_EVIDENCE:"):
            result["missing_evidence"] = line.replace("MISSING_EVIDENCE:", "").strip()

    return result


# ------------------------------------------------------------------
# Final output formatter
# ------------------------------------------------------------------
def format_output(query: str, routing: dict, answer: str, critique: dict, chunks: list[dict]) -> str:
    sources = list({c["source"] for c in chunks})
    verdict = critique["verdict"]
    verdict_symbol = {"Fully Answered": "✓", "Partially Answered": "~", "Not Answered": "✗"}.get(verdict, "?")

    lines = [
        "=" * 60,
        "  AUDIT RESPONSE",
        "=" * 60,
        f"Question    : {query}",
        f"Keywords    : {', '.join(routing['keywords'])}",
        f"Doc Type    : {routing['doc_type_hint']}",
        "-" * 60,
        "Answer:",
        answer,
        "-" * 60,
        f"Coverage    : {verdict_symbol} {verdict}  (confidence: {critique['confidence']:.0%})",
        f"Sources     : {', '.join(sources) if sources else 'None'}",
        f"Missing     : {critique['missing_evidence']}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------
# Main pipeline orchestration
# ------------------------------------------------------------------
def run_pipeline(query: str, retrieval_agent: RetrievalAgent) -> str:
    print("\n[1/4] Routing Agent — analyzing question...")
    routing = routing_agent(query)
    print(f"      Refined query : {routing['refined_query']}")
    print(f"      Keywords      : {routing['keywords']}")

    print("[2/4] Retrieval Agent — searching documents...")
    chunks = retrieval_agent.retrieve(routing["refined_query"], top_k=TOP_K)
    print(f"      Found {len(chunks)} relevant chunks.")

    print("[3/4] Synthesis Agent — generating answer...")
    answer = synthesis_agent(query, chunks)

    print("[4/4] Critic Agent — verifying answer...\n")
    critique = critic_agent(query, answer, chunks)

    return format_output(query, routing, answer, critique, chunks)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
def main():
    if not os.path.isdir(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER)
        print(f"[Setup] Created docs/ folder. Place PDF files there and restart.")
        return

    print("[Setup] Loading documents and building index...")
    agent = RetrievalAgent(docs_folder=DOCS_FOLDER)

    print("\nAudit RAG Pipeline — Ready")
    print("Type your audit question. Enter 'quit' to exit.\n")

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

        output = run_pipeline(query, agent)
        print(output)


if __name__ == "__main__":
    main()
