import os
import uuid
import json
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from compressor.pruner import greedy_phrase_prune, phrase_surprisals_for_segment
from compressor.hybrid import hybrid_compress
from models.embedder import Embedder
from models.scorer import Scorer
from models.summarizer import Summarizer
from compressor.utils import rule_compress, cleanup_text_t5

# Load environment variables from .env file
load_dotenv()

# Optional: Gemini client check
try:
    from google import genai
except Exception:
    genai = None

try:
    from sentence_transformers import util
except Exception:
    util = None

app = Flask(__name__, template_folder="templates")
app.config.from_pyfile("instance/config.py", silent=True)

# Load models (singleton)
print("⏳ Loading Models... (This may take a moment)")
SCORER = Scorer()            # GPT-2 scorer
EMBEDDER = Embedder()        # SentenceTransformer
SUMMARIZER = Summarizer()    # DistilBART/T5
print("✅ Models Loaded.")

# Simple in-memory run cache
RUNS = {}

## ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json or {}
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    # compute phrase surprisal candidates
    words, spans, phrase_scores, protected = phrase_surprisals_for_segment(prompt)
    
    # FIX: phrase_scores contains 'set' objects which break JSON. 
    # We clean them up for the frontend.
    clean_candidates = []
    for cand in phrase_scores[:200]:
        clean_candidates.append({
            "phrase": cand["phrase"],
            "norm": cand["norm"]
            # We explicitly exclude 'indices' (the set) here
        })

    token_count = len(SCORER.tokenizer.tokenize(prompt))
    
    resp = {
        "words": words,
        "word_spans": spans,
        "phrase_candidates_count": len(phrase_scores),
        "top_phrase_candidates": clean_candidates, # Use the clean list
        "token_count": token_count
    }
    return jsonify(resp)

@app.route("/api/prune", methods=["POST"])
def api_prune():
    data = request.json or {}
    prompt = data.get("prompt", "")
    keep_ratio = float(data.get("keep_ratio", 0.85))
    max_phrase_len = int(data.get("max_phrase_len", 4))
    sim_thresh = float(data.get("sim_threshold", 0.82))

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    compressed, sim, log = greedy_phrase_prune(
        prompt,
        keep_ratio=keep_ratio,
        max_phrase_len=max_phrase_len,
        similarity_threshold=sim_thresh,
        scorer=SCORER,
        embedder=EMBEDDER
    )

    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "original": prompt,
        "compressed": compressed,
        "sim": sim,
        "log": log
    }
    return jsonify({"run_id": run_id, "compressed": compressed, "sim": sim, "log_len": len(log), "log": log})

@app.route("/api/hybrid", methods=["POST"])
def api_hybrid():
    data = request.json or {}
    prompt = data.get("prompt", "")
    sim_thresh = float(data.get("sim_threshold", 0.82))
    keep_ratio_phrases = float(data.get("keep_ratio_phrases", 0.6))

    if not prompt:
        return jsonify({"error": "prompt required"}), 400

    # run hybrid compress
    compressed = hybrid_compress(prompt,
                                 summarizer=SUMMARIZER,
                                 embedder=EMBEDDER,
                                 scorer=SCORER,
                                 sim_thresh=sim_thresh,
                                 keep_ratio_phrases=keep_ratio_phrases)

    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "original": prompt,
        "hybrid": compressed
    }
    return jsonify({"run_id": run_id, "hybrid": compressed})

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.json or {}
    which = data.get("which", "original")  # original|pruned|hybrid
    run_id = data.get("run_id")
    model = data.get("model", "gemini-2.5-flash")
    prompt_inline = data.get("prompt")
    
    prompt_text = ""
    if run_id:
        run = RUNS.get(run_id)
        if not run:
            return jsonify({"error": "run_id not found"}), 404
        if which == "original":
            prompt_text = run.get("original")
        elif which == "pruned":
            prompt_text = run.get("compressed")
        else:
            prompt_text = run.get("hybrid") or run.get("compressed") or run.get("original")
    else:
        if not prompt_inline:
            return jsonify({"error": "either run_id or prompt is required"}), 400
        prompt_text = prompt_inline

    # Option: local mock if genai not available
    if genai is None:
        return jsonify({"mock": True, "text": f"(mock) Output for prompt length {len(prompt_text)} chars"}), 200

    # Build client
    api_key = os.environ.get("GOOGLE_API_KEY") or app.config.get("GOOGLE_API_KEY")
    if not api_key:
        return jsonify({"error": "GOOGLE_API_KEY missing"}), 500
        
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt_text)
        return jsonify({"text": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/validate", methods=["POST"])
def api_validate():
    data = request.json or {}
    a = data.get("a", "")
    b = data.get("b", "")
    if not a or not b:
        return jsonify({"error": "both 'a' and 'b' required"}), 400
    if not util:
        return jsonify({"error": "sentence_transformers not available"}), 500
    emb_a = EMBEDDER.encode(a, convert_to_tensor=True)
    emb_b = EMBEDDER.encode(b, convert_to_tensor=True)
    score = float(util.pytorch_cos_sim(emb_a, emb_b).item())
    ta = len(SCORER.tokenizer.tokenize(a))
    tb = len(SCORER.tokenizer.tokenize(b))
    return jsonify({"similarity": score, "tokens_a": ta, "tokens_b": tb})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)