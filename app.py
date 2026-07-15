"""
Local web app for answer_question.py: type an audit question in the browser,
get the synthesized answer streamed back as it's generated.

Usage: python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import json

from flask import Flask, Response, jsonify, render_template, request

from answer_question import run_question_stream
from config import LATEST_REVISIONS_CSV
from retrieval_agent import RetrievalAgent
from utils import describe_error

app = Flask(__name__)

print("[app] Building RetrievalAgent (connecting to Pinecone)...")
try:
    agent = RetrievalAgent(docs_folder="docs", csv_path=LATEST_REVISIONS_CSV)
except Exception as e:
    print(f"[app] STARTUP FAILED: {describe_error(e)}")
    raise
print("[app] Ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    def generate():
        try:
            for event in run_question_stream(agent, question):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            error_msg = describe_error(e)
            print(f"  [/ask error] {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
