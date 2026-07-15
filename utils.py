import csv
import os
import time

from openai import APIError, OpenAI
from pinecone.errors.exceptions import ApiError as PineconeApiError

from config import LLM_PROVIDER, OPENAI_BASE_URL

LLM_RETRIES = 2  # attempts on transient OpenRouter/OpenAI errors (connection, rate
                 # limit, timeout -- all subclasses of openai.APIError) before giving up
LLM_RETRY_BASE_WAIT = 2  # seconds; doubles each retry (exponential backoff)

CHUNK_CSV_FIELDNAMES = ["sub_query", "rerank_score", "score", "source", "document_id",
                        "revision", "chunk_index", "title", "page", "text_content"]


def retry_with_backoff(fn, *, retries: int, base_wait: float, exceptions, label: str):
    """Call fn() (a zero-arg callable), retrying up to `retries` attempts total on
    `exceptions` with exponential backoff (base_wait * 2**attempt), before re-raising
    on the final attempt. Returns fn()'s result on success."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except exceptions:
            if attempt == retries:
                raise
            wait_s = base_wait * (2 ** (attempt - 1))
            print(f"  [{label}] retrying in {wait_s}s (attempt {attempt}/{retries})...")
            time.sleep(wait_s)


def describe_error(e: Exception) -> str:
    """Label an exception by which external service it actually came from
    (Pinecone vs. the LLM provider), so an error surfaced to logs/the UI is
    immediately actionable instead of a bare exception string. Also flags the
    specific "redirected to a proxy gateway" pattern we've hit ourselves --
    a 307 redirect to a domain like gateway.zscaler.net means a corporate
    security proxy is intercepting the request before it reaches the real
    service at all, which reads very differently from a real Pinecone/LLM
    failure and is worth calling out explicitly."""
    msg = str(e)
    if isinstance(e, PineconeApiError):
        label = "Pinecone error"
    elif isinstance(e, APIError):
        label = "LLM error"
    else:
        label = "Error"

    if "307" in msg and ("zscaler" in msg.lower() or "gateway" in msg.lower()):
        return (f"[{label}] {msg} -- this looks like a proxy/security gateway "
                f"(e.g. Zscaler) intercepting the request before it reached the "
                f"real service. Check network/VPN connectivity, not just the "
                f"service itself.")
    return f"[{label}] {msg}"


def get_openai_client() -> OpenAI:
    if LLM_PROVIDER == "ollama":
        # Ollama's OpenAI-compatible endpoint ignores the key, but the SDK requires
        # a non-empty string to construct the client.
        return OpenAI(api_key="ollama", base_url=OPENAI_BASE_URL)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)


def call_llm(client: OpenAI, model: str, system_prompt: str, user_prompt: str, *,
             retries: int = LLM_RETRIES, base_wait: float = LLM_RETRY_BASE_WAIT, label: str = "LLM error",
             max_tokens: int | None = None):
    """chat.completions.create(), retrying with backoff on openai.APIError.

    `max_tokens` is meant as a generous safety backstop (well above what a
    properly concise answer needs), not the primary length control -- prompting
    the model to be concise is the real lever; this just caps runaway output."""
    return retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        ),
        retries=retries, base_wait=base_wait, exceptions=APIError, label=label,
    )


def call_llm_stream(client: OpenAI, model: str, system_prompt: str, user_prompt: str, *,
                     retries: int = LLM_RETRIES, base_wait: float = LLM_RETRY_BASE_WAIT, label: str = "LLM error",
                     max_tokens: int | None = None):
    """Like call_llm, but yields text deltas as they arrive instead of returning the
    full response at once. Retry-with-backoff covers establishing the stream; once
    streaming begins, a mid-stream failure propagates to the caller as-is (there's
    no clean way to "retry" after some tokens have already been yielded)."""
    stream = retry_with_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        ),
        retries=retries, base_wait=base_wait, exceptions=APIError, label=label,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_dicts_to_csv(path: str, rows: list[dict], fieldnames: list[str]):
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_answer_file(path: str, question: str, answer: str, chunks_used, extra_lines: list[str] | None = None):
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Audit question: {question}\n")
        for line in extra_lines or []:
            f.write(line + "\n")
        f.write(f"Chunks used: {chunks_used}\n")
        f.write("=" * 80 + "\n")
        f.write(answer + "\n")


def chunk_key(c: dict) -> tuple:
    return (c.get("document_id"), c.get("revision"), c.get("chunk_index"))


def chunk_from_metadata(md: dict) -> dict:
    return {
        "source": md.get("source"),
        "document_id": md.get("document_id"),
        "revision": md.get("revision"),
        "chunk_index": md.get("chunk_index"),
        "title": md.get("title"),
        "heading_level": md.get("heading_level"),
        "page": md.get("page"),
        "text_content": md.get("text_content"),
        "folder_type": md.get("folder_type"),
        "subfolder": md.get("subfolder"),
    }
