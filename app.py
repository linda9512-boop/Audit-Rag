"""
Local web app for answer_question.py: type an audit question in the browser,
get the synthesized answer streamed back as it's generated.

Usage: python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import json
import time

from flask import Flask, Response, jsonify, make_response, render_template, request

from answer_question import run_question_stream
from config import LATEST_REVISIONS_CSV
from retrieval_agent import RetrievalAgent
from utils import describe_error

app = Flask(__name__)

print("[app] Building RetrievalAgent (connecting to Pinecone)...")
_t0 = time.perf_counter()
try:
    agent = RetrievalAgent(docs_folder="docs", csv_path=LATEST_REVISIONS_CSV)
except Exception as e:
    print(f"[app] STARTUP FAILED: {describe_error(e)}")
    raise
print(f"[app] Ready. [timing] startup: {time.perf_counter() - _t0:.2f}s")


@app.route("/")
def index():
    # This page changes often during development -- tell the browser never to
    # reuse a cached copy, so a normal refresh always shows the latest version
    # instead of requiring a hard-refresh to bypass the browser's own cache.
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    def generate():
        t_req = time.perf_counter()
        print(f"[/ask] question: {question!r}")
        try:
            for event in run_question_stream(agent, question):
                if event.get("type") == "done":
                    print(f"[/ask] [timing] total request: {time.perf_counter() - t_req:.2f}s")
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_msg = describe_error(e)
            print(f"  [/ask error] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=False, port=5001)
