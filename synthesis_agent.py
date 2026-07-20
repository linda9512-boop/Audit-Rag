import csv
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from config import OPENAI_MODEL, OUTPUT_DIR
from timing_log import tlog
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
   it to just the document ID. i Always use this exact format, every time, even
   when citing several sources together: square brackets, the word "page" (not
   "p." or "pg"), and one [source, page] per citation rather than combining
   multiple sources inside a single bracket. For example, write
   "[D54978 HAL 4.0 Regulatory Plan Rev 5.0.pdf, page 5] [D58308 HAL 4.0 Device
   Description Rev 1.0.docx.pdf, page 3]" -- not "(D54978, p. 5; D58308, p. 3)"
   or any other abbreviated or combined form.
3. When an excerpt contains a device-specific determination -- a classification
   outcome, a rule-by-rule justification, a test result, a decision made for
   Halcyon 4.0 itself -- state that concrete determination. Do not stop at
   restating the general regulation or rule text if the excerpt also shows how
   it was applied to Halcyon 4.0.
4. If the excerpts only partially answer the question, say plainly what's missing --
   do not guess or fill gaps with outside knowledge.
5. If none of the excerpts are relevant to the audit question, say plainly that
   no relevant evidence was found -- do not force an answer out of irrelevant material.
6. Ignore excerpts that are irrelevant boilerplate (cover pages, unrelated sections)
   even if they were retrieved.
7. ONLY USE DOCUMENTS THAT DIRECTLY ANSWER THE AUDIT QUESTION. Avoid documents
   that only provide supporting, reference, or background information -- for
   each document you cite, you should be able to explain how it directly
   answers the main query, not just that it's topically related. Do not
   stretch an excerpt to fit a part of the question it doesn't actually
   address, just because it shares keywords or a general topic -- if nothing
   retrieved actually addresses that part, say so instead of forcing a
   tangentially-related excerpt to answer it.

   Example -- Question: "Are all devices, including accessories, software,
   variants, and families, appropriately identified and included?" 
   Retrieved evidence available: Accessory and Component List, Device Description,
   Validation Summary Report, Test System Configuration List (TSCL).
   Good: use the Accessory and Component List, Device Description, and
   Validation Summary Report, because they directly identify the released
   device, accessories, and software.
   Bad: do not use the Test System Configuration List (TSCL) as evidence for
   software identification -- TSCL is just the environment setup of the
   selected test cell for validation activities; it doesn't reflect the
   device's latest software version, so it's background/supporting
   information, not a direct answer.
8. Watch the tense and status of what an excerpt actually says. Action items,
   findings, meeting notes, and plans ("shall define," "need to," "is planned,"
   "recommended," open CAPAs) describe something that does NOT yet exist or
   isn't yet resolved -- do not report these as an established, existing process
   or outcome. Only present something as in place if the excerpt describes it as
   already implemented, approved, or completed.
9. Be concise. For each document (or point) you cite, state the determination plainly with its
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
    t0 = time.perf_counter()
    context = build_context(chunks)
    tlog(f"  [synthesis] build_context ({len(chunks)} chunks, {len(context)} chars): {time.perf_counter() - t0:.3f}s")

    user_prompt = (
        f"Audit question: {question}\n\n"
        f"Retrieved document excerpts:\n\n{context}\n\n"
        "Answer the audit question using only the excerpts above, citing sources."
    )
    tlog(f"  [synthesis] user_prompt length: {len(user_prompt)} chars")

    t0 = time.perf_counter()
    response = call_llm(client, model, SYSTEM_PROMPT, user_prompt, label="synthesis LLM error",
                         max_tokens=MAX_ANSWER_TOKENS, temperature=0)
    tlog(f"  [synthesis] LLM call: {time.perf_counter() - t0:.2f}s")
    return response.choices[0].message.content


def synthesize_stream(question: str, chunks: list[dict], model: str = OPENAI_MODEL):
    """Like synthesize(), but yields the answer text as it's generated instead of
    returning the full string once complete."""
    client = get_openai_client()
    t0 = time.perf_counter()
    context = build_context(chunks)
    tlog(f"  [synthesis] build_context ({len(chunks)} chunks, {len(context)} chars): {time.perf_counter() - t0:.3f}s")

    user_prompt = (
        f"Audit question: {question}\n\n"
        f"Retrieved document excerpts:\n\n{context}\n\n"
        "Answer the audit question using only the excerpts above, citing sources."
    )
    tlog(f"  [synthesis] user_prompt length: {len(user_prompt)} chars")

    t0 = time.perf_counter()
    first_token = True
    for delta in call_llm_stream(client, model, SYSTEM_PROMPT, user_prompt, label="synthesis LLM error",
                                  max_tokens=MAX_ANSWER_TOKENS, temperature=0):
        if first_token:
            tlog(f"  [synthesis] time to first token: {time.perf_counter() - t0:.2f}s")
            first_token = False
        yield delta


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
