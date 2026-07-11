"""
Generates facet-specific sub-queries for a broad audit question using an LLM, to improve retrieval of narrowly-scoped documents.

"""
import json
import os
import time

from dotenv import load_dotenv
from openai import APIError, OpenAI

from synthesis_agent import OPENROUTER_BASE_URL, OPENAI_MODEL

load_dotenv()

LLM_RETRIES = 2  # attempts on transient OpenRouter/OpenAI errors (connection, rate
                 # limit, timeout -- all subclasses of openai.APIError) before giving up
LLM_RETRY_BASE_WAIT = 2  # seconds; doubles each retry (exponential backoff)

SYSTEM_PROMPT = """You are helping decompose a broad medical device audit question into
narrower search queries for a semantic search + reranking retrieval system.

A single broad question often embeds too generically to surface narrowly-scoped
documents -- e.g. a question naming several distinct things ("accessories, software,
variants, and families") will match generic compliance-checklist language far more
strongly than a document that's actually just a bare accessory list or a product
variant spec, because the checklist repeats more of the question's own vocabulary.

Given an audit question, produce 1-4 sub-queries that each target ONE distinct
facet or concrete evidence type the question is really asking about. 
if the question is already very narrow, you may return just one sub-query that is a rephrasing of the original question.1
Each sub-query should:
  - use concrete, document-like phrasing (the kind of words that would appear in
    the title or opening line of the actual source document you're looking for),
    not abstract audit-speak
  - target a genuinely different facet than the other sub-queries, not a rephrasing
    of the same one
  - stay specific enough to distinguish itself from the other facets, but not so
    narrow it only matches one exact phrase

Example:
  Question: "Are all devices, including accessories, software, variants, and
    families, appropriately identified and included?"
  Sub-queries: ["accessory and component list for the device",
    "software identification, version control, and configuration management",
    "device variants, models, and product family configurations"]

Respond with a JSON array of strings only, e.g. ["sub-query 1", "sub-query 2", "sub-query 3"].
No other text.
"""


def generate_subqueries(question: str, max_n: int = 4, model: str = OPENAI_MODEL) -> list[str]:
    """Lets the LLM decide freely how many sub-queries a question actually needs
    (1-4, per SYSTEM_PROMPT) rather than being told to hit a specific target.
    `max_n` is only a defensive cap in case the model returns more than instructed.

    Falls back to [question] itself (i.e. no facet split) if the call keeps
    failing or the model's response isn't parseable -- a missing decomposition
    just means the caller retrieves on the original question directly, which is
    recoverable, so this never raises for that reason. A genuine failure to reach
    the API at all (after retries) still raises, since there's no way to fall
    back on real API access being unavailable.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=OPENROUTER_BASE_URL)

    user_prompt = f"Audit question: {question}"

    response = None
    for attempt in range(1, LLM_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            break
        except APIError:
            if attempt == LLM_RETRIES:
                raise
            wait_s = LLM_RETRY_BASE_WAIT * (2 ** (attempt - 1))
            print(f"  [subquery LLM error] retrying in {wait_s}s "
                  f"(attempt {attempt}/{LLM_RETRIES})...")
            time.sleep(wait_s)

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
