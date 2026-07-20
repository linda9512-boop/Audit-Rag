import os

from dotenv import load_dotenv

load_dotenv()  # loaded here (not left to each importing script) so LLM_PROVIDER etc.
               # below are populated regardless of when the importer calls load_dotenv()

# LLM_PROVIDER selects which OpenAI-compatible endpoint the whole pipeline talks to:
#   "openrouter" (default) -- OPENAI_API_KEY required, hosted models
#   "ollama"               -- points at a local/LAN Ollama server instead, no real API key needed
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").lower()

if LLM_PROVIDER == "ollama":
    OPENAI_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://10.1.226.105:11434/v1")
    OPENAI_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b-a4b-it-q8_0")
else:
    OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    OPENAI_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")  # OpenRouter model id (provider/model)

PINECONE_INDEX_NAME = "audit"

# Per-request timing breakdown (see timing_log.py) -- prints and logs to outputs/question_runs/timing.log.
# Set TIMING_ENABLED=false in production to silence it without touching any of the tlog() call sites.
TIMING_ENABLED = os.environ.get("TIMING_ENABLED", "true").lower() in ("1", "true", "yes")

OUTPUT_DIR = "outputs"  # root folder for all generated run artifacts (gitignored)
LATEST_REVISIONS_CSV = "outputs/latest_revisions.csv"  # written by extracting_latest.py,
                                                        # read by every script that builds a RetrievalAgent

# Retrieval + rerank sizing for answer_question.py's per-sub-query retrieval.
ANSWER_CANDIDATE_POOL = 100  # raw candidates fetched by embedding search, before reranking --
                             # Pinecone's bge-reranker-v2-m3 hard-caps at 100 documents per
                             # rerank() call, so this can't go higher without batching the
                             # rerank call itself.
ANSWER_TOP_N_PER_SUBQUERY = 30  # kept after reranking, per sub-query

# Hard cap on augment_chunks()'s output size. 
MAX_TOTAL_CHUNKS = 100000
