import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from config import OPENAI_MODEL, OUTPUT_DIR
from utils import call_llm, call_llm_stream, get_openai_client, write_answer_file

load_dotenv()

MAX_ANSWER_TOKENS = 3000  # generous safety backstop, not the primary length control --
                          # SYSTEM_PROMPT rule 11 (conciseness) is what actually shapes
                          # typical answer length; this only guards against runaway output

SYSTEM_PROMPT = """You are an audit assistant for a medical device Design History File.

The subject device under audit is the Halcyon 4.0 (HAL 4.0) radiotherapy system --
every document in this corpus concerns this specific device, not devices in general.

You will be given an audit question and a set of retrieved document excerpts (each
tagged with its source -- folder path and filename -- page, and section title --
plus a Role). The Role tells you why each excerpt is here:
  - "matched": the search engine judged this excerpt itself relevant to the question.
    This is your primary evidence.
  - "neighbor": the section immediately before/after a matched excerpt (same page or
    the page before/after). Use it for the same purpose as a "matched" excerpt -- if
    it genuinely provides evidence for the audit question, cite it like any other
    excerpt -- and also use it to correctly interpret an adjacent matched excerpt
    (e.g. supplying a lead-in sentence a truncated table needs). The Role label is
    for your own traceability, not a restriction on what you're allowed to use.
  - "header": that document's own title page / approval / revision history, included
    only so you know what the document actually is (e.g. an initial release vs. a
    later revision). Do not cite it as evidence for the audit question itself.
Using only the excerpts provided:

1. Your primary goal is to identify which documents should be submitted as
   evidence for this audit question -- the explanation below exists to justify
   that list, not the other way around.
2. Cite which document(s) support each part of your answer, as [source, page],
   using the filename exactly as given in each excerpt's tag -- do not shorten
   it to just the document ID.
3. When an excerpt contains a device-specific determination -- a classification
   outcome, a rule-by-rule justification, a test result, a decision made for
   Halcyon 4.0 itself -- state that concrete determination. Do not stop at
   restating the general regulation or rule text if the excerpt also shows how
   it was applied to Halcyon 4.0.
4. If two excerpts conflict (e.g. different revisions of the same or related
   documents state different things), do not silently pick one -- report both
   and note that they conflict.
5. If the excerpts only partially answer the question, say plainly what's missing --
   do not guess or fill gaps with outside knowledge.
6. If none of the excerpts are relevant to the audit question, say plainly that
   no relevant evidence was found -- do not force an answer out of irrelevant material.
7. Ignore excerpts that are irrelevant boilerplate (cover pages, unrelated sections)
   even if they were retrieved.
8. Do not stretch an excerpt to fit a part of the question it doesn't actually
   address, just because it shares keywords or a general topic. Cite an excerpt
   only if it genuinely provides evidence for the specific thing being asked --
   if nothing retrieved actually addresses that part, say so instead of forcing
   a tangentially-related excerpt to answer it.
9. Watch the tense and status of what an excerpt actually says. Action items,
   findings, meeting notes, and plans ("shall define," "need to," "is planned,"
   "recommended," open CAPAs) describe something that does NOT yet exist or
   isn't yet resolved -- do not report these as an established, existing process
   or outcome. Only present something as in place if the excerpt describes it as
   already implemented, approved, or completed.
10. End your answer with a "Documents cited:" list -- every [source, page] you
    actually cited above, in the same [source, page] format used inline, one per
    line, deduplicated (the same source+page pair listed once even if referenced
    more than once in the body above). If a source was cited at more than one
    page, list each distinct page separately.
11. Be concise. For each document (or point) you cite, keep the description of
    what it shows to under 100 words -- state the determination plainly with its
    citation and move on, rather than elaborating at length. This applies per
    citation, not to the answer as a whole, so a question needing many citations
    still gets a complete answer -- it's each individual explanation that stays
    tight, not the total count of documents covered. Do not restate the question,
    re-explain what a Role label means, or pad the answer with summary/recap
    sections.
"""


def load_chunks(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        role = c.get("context_role") or "matched"
        tag = f"[Role: {role} | Source: {c['source']} | Page: {c['page']} | Section: {c['title']}]"
        parts.append(f"{tag}\n{c['text_content']}")
    return "\n\n---\n\n".join(parts)


def synthesize(question: str, chunks: list[dict], model: str = OPENAI_MODEL) -> str:
    client = get_openai_client()
    context = build_context(chunks)

    user_prompt = (
        f"Audit question: {question}\n\n"
        f"Retrieved document excerpts:\n\n{context}\n\n"
        "Answer the audit question using only the excerpts above, citing sources."
    )

    response = call_llm(client, model, SYSTEM_PROMPT, user_prompt, label="synthesis LLM error",
                         max_tokens=MAX_ANSWER_TOKENS)
    return response.choices[0].message.content


def synthesize_stream(question: str, chunks: list[dict], model: str = OPENAI_MODEL):
    """Like synthesize(), but yields the answer text as it's generated instead of
    returning the full string once complete."""
    client = get_openai_client()
    context = build_context(chunks)

    user_prompt = (
        f"Audit question: {question}\n\n"
        f"Retrieved document excerpts:\n\n{context}\n\n"
        "Answer the audit question using only the excerpts above, citing sources."
    )

    yield from call_llm_stream(client, model, SYSTEM_PROMPT, user_prompt, label="synthesis LLM error",
                                max_tokens=MAX_ANSWER_TOKENS)


OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthesis_answer.txt")

if __name__ == "__main__":
    MAIN_QUERY = (
        "Verify that those devices that are, by regulation, subject to design and "
        "development procedures have been identified. (See Annex 1)"
    )

    chunks = load_chunks(os.path.join(OUTPUT_DIR, "retrieved_documents.csv"))
    print(f"Synthesizing answer from {len(chunks)} retrieved chunks...\n")

    answer = synthesize(MAIN_QUERY, chunks)
    print("=" * 80)
    print(answer)

    write_answer_file(OUTPUT_FILE, MAIN_QUERY, answer, len(chunks))
    print(f"\nSaved answer to {OUTPUT_FILE}")
