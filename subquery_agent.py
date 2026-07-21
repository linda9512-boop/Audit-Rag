"""
Generates facet-specific sub-queries for a broad audit question using an LLM, to improve retrieval of narrowly-scoped documents.

"""
import json

from dotenv import load_dotenv

from config import OPENAI_MODEL
from utils import call_llm, get_openai_client

load_dotenv()

SYSTEM_PROMPT = """You are helping decompose a broad medical device audit question into
narrower search queries for a semantic search + reranking retrieval system.

A single broad question often embeds too generically to surface narrowly-scoped
documents -- e.g. a question naming several distinct things ("accessories, software,
variants, and families") will match generic compliance-checklist language far more
strongly than a document that's actually just a bare accessory list or a product
variant spec, because the checklist repeats more of the question's own vocabulary.

The same problem repeats one level down: a sub-query that itself joins several distinct
concepts with "and" or a comma (e.g. "software identification, version control, and
configuration management") also scores poorly against any one of those concepts' real
source documents. The reranker scores literal phrase overlap, so a document that is
purely a software identification/inventory list scores worse against that joined phrase
than against "software identification" alone, even though it's exactly the right document.

Given an audit question, first identify every distinct, concrete keypoint it is really
asking about, then produce one short sub-query per keypoint -- 1-5 sub-queries total.
If the question is already narrow enough to be one keypoint, return just one sub-query
that is a rephrasing of the original question.

Never join multiple keypoints into one sub-query with "and" or a comma. If what looks
like a single topic actually covers more than one kind of evidence (e.g. "software"
implying both identification/inventory AND configuration management as two different
kinds of documents), split it into that many separate sub-queries instead of one
combined one. Conversely, don't split things that are genuinely the same kind of
evidence just to hit a higher count (e.g. device names, models, and families are
typically identified together in the same document, so they can stay one sub-query).

Each sub-query should:
  - be short (2-6 words) and use concrete, document-like phrasing (the kind of words
    that would appear in the title or opening line of the actual source document you're
    looking for), not abstract audit-speak
  - target exactly ONE keypoint -- never multiple concepts joined by "and"/commas
    unless those concepts are always documented together in practice
  - stay specific enough to distinguish itself from the other sub-queries, but not so
    narrow it only matches one exact phrase

Example:
  Question: "Are all devices, including accessories, software, variants, and
    families, appropriately identified and included?"
  Sub-queries: ["device names, models, and families",
    "included accessories",
    "software identification",
    "software configuration management procedure",
    "product variants and configurations"]

Respond with a JSON array of strings only, e.g. ["sub-query 1", "sub-query 2", "sub-query 3"].
No other text.
"""


def generate_subqueries(question: str, max_n: int = 5, model: str = OPENAI_MODEL) -> list[str]:
    """Lets the LLM decide freely how many sub-queries a question actually needs
    (1-5, per SYSTEM_PROMPT) rather than being told to hit a specific target.
    `max_n` is only a defensive cap in case the model returns more than instructed.

    Falls back to [question] itself (i.e. no facet split) if the call keeps
    failing or the model's response isn't parseable -- a missing decomposition
    just means the caller retrieves on the original question directly, which is
    recoverable, so this never raises for that reason. A genuine failure to reach
    the API at all (after retries) still raises, since there's no way to fall
    back on real API access being unavailable.
    """
    client = get_openai_client()

    user_prompt = f"Audit question: {question}"

    response = call_llm(client, model, SYSTEM_PROMPT, user_prompt, label="subquery LLM error", temperature=0)

    content = response.choices[0].message.content.strip()
    # Strip markdown code fences if the model wrapped the JSON in ```...```
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        subqueries = json.loads(content)
        if not isinstance(subqueries, list) or not all(isinstance(s, str) for s in subqueries) or not subqueries:
            raise ValueError(f"Expected a non-empty JSON array of strings, got: {content!r}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [subquery parse error] falling back to the original question unsplit: {e}")
        return [question]

    return subqueries[:max_n]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    test_question = "Are all devices, including accessories, software, variants, and families, appropriately identified and included?"
    for q in generate_subqueries(test_question):
        print(f"  - {q}")
