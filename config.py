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

OUTPUT_DIR = "outputs"  # root folder for all generated run artifacts (gitignored)
LATEST_REVISIONS_CSV = "outputs/latest_revisions.csv"  # written by extracting_latest.py,
                                                        # read by every script that builds a RetrievalAgent

# Retrieval + rerank sizing for answer_question.py's per-sub-query retrieval.
ANSWER_CANDIDATE_POOL = 65  # raw candidates fetched by embedding search, before reranking
ANSWER_TOP_N_PER_SUBQUERY = 30  # kept after reranking, per sub-query

# Hard cap on augment_chunks()'s output size. Extensively benchmarked against a
# token-based budget (100k down to 200 tokens) -- context size never showed a
# consistent effect on synthesize() latency (that's driven by output length, not
# input size -- see SYSTEM_PROMPT rule 11 in synthesis_agent.py for the real fix).
# Reverted to a plain chunk count for simplicity now that the token-precision
# this used to buy isn't earning its complexity.
MAX_TOTAL_CHUNKS = 200
