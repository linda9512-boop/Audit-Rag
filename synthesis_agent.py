import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENAI_MODEL = "openai/gpt-4o"  # OpenRouter model id (provider/model)

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
   using the full source exactly as given in each excerpt's tag (including its
   folder path) -- do not shorten it to just the document ID.
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
10. End your answer with a "Documents cited:" list -- every source you actually
    cited above, deduplicated, one per line.
"""


def load_chunks(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        folder = c.get("folder_type") or ""
        subfolder = c.get("subfolder") or ""
        folder_path = " / ".join(p for p in (folder, subfolder) if p)
        source = f"{folder_path} / {c['source']}" if folder_path else c["source"]
        role = c.get("context_role") or "matched"
        tag = f"[Role: {role} | Source: {source} | Page: {c['page']} | Section: {c['title']}]"
        parts.append(f"{tag}\n{c['text_content']}")
    return "\n\n---\n\n".join(parts)


def synthesize(question: str, chunks: list[dict], model: str = OPENAI_MODEL) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=OPENROUTER_BASE_URL)
    context = build_context(chunks)

    user_prompt = (
        f"Audit question: {question}\n\n"
        f"Retrieved document excerpts:\n\n{context}\n\n"
        "Answer the audit question using only the excerpts above, citing sources."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


OUTPUT_FILE = "synthesis_answer.txt"

if __name__ == "__main__":
    MAIN_QUERY = (
        "Verify that those devices that are, by regulation, subject to design and "
        "development procedures have been identified. (See Annex 1)"
    )

    chunks = load_chunks("retrieved_documents.csv")
    print(f"Synthesizing answer from {len(chunks)} retrieved chunks...\n")

    answer = synthesize(MAIN_QUERY, chunks)
    print("=" * 80)
    print(answer)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Audit question: {MAIN_QUERY}\n")
        f.write(f"Chunks used: {len(chunks)}\n")
        f.write("=" * 80 + "\n")
        f.write(answer + "\n")

    print(f"\nSaved answer to {OUTPUT_FILE}")
