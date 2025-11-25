# 🎯 Prompt Compression Visualizer

An interactive, end-to-end system to analyze, compress, visualize, and validate large prompts used by LLMs. This tool bridges information theory (surprisal), NLP compression techniques, and LLM generation, giving users full transparency into how prompt compression algorithms work and how compression affects output quality.

**Live Demo:** Run the Flask backend and open the web UI to start compressing prompts instantly.

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Algorithms](#algorithms)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## ❓ Problem Statement

Large prompts sent to LLMs are expensive and inefficient:

- **Increased Latency**: More tokens = slower response times
- **Higher Costs**: Most LLM APIs charge per token
- **Token Limits**: Many models have hard limits on input length
- **Redundancy**: Prompts often contain repetitive, low-information content

**Solution**: Automatically compress prompts while preserving:
- ✅ Semantic meaning
- ✅ Instruction-following ability
- ✅ Structural integrity (examples, formatting)
- ✅ LLM output quality

---

## ✨ Features

### 🔵 Phrase-Level Surprisal-Based Greedy Pruning
- **Information-theoretic approach** using GPT-2 token logprobs
- **Surprisal scoring**: Identifies low-information, redundant phrases
- **Iterative removal**: Greedily removes lowest-surprisal phrases until target compression ratio
- **Semantic safeguards**: Validates compressed prompt maintains similarity (default: 0.82)
- **Structural protection**: Preserves labels, examples, and critical grammar

### 🟣 Hybrid Compression Pipeline
- **Multi-stage compression**: Combines rule-based, summarization, and pruning techniques
- **Rule-based simplification**: Cleans up bullet points, repeated adjectives, formatting
- **Summarization fallback**: Uses DistilBART to condense lengthy sections
- **Semantic validation**: Ensures summarizations preserve meaning
- **Aggressive compression**: Achieves 25–40% reduction with minimal quality loss

### 📊 Interactive Web UI
- **Modern, responsive design** with tabbed output viewer
- **Real-time analysis**: See phrase candidates and surprisal distribution
- **Side-by-side comparison**: View original vs. compressed prompts and LLM outputs
- **Compression metrics**: Track token counts and semantic similarity
- **Iteration log**: Follow the compression process step-by-step
- **JSON export**: Download full compression run data for analysis

### 🤖 LLM Integration
- **Gemini API support**: Generate outputs with compressed prompts
- **Mock mode**: Test without API keys
- **Semantic similarity validation**: Compare outputs using SentenceTransformer embeddings

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI (HTML/JS/CSS)                     │
│  - Prompt input, controls, visualization, tabbed outputs    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Flask Backend (app.py)                          │
│  - 5 REST APIs: /analyze, /prune, /hybrid, /generate, ...   │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼───┐      ┌───▼────┐    ┌───▼────┐
    │Pruner │      │Hybrid  │    │Models  │
    │Module │      │Module  │    │Module  │
    └───┬───┘      └───┬────┘    └───┬────┘
        │              │             │
        │         ┌────┴─────┬───────┴─────┐
        │         │          │             │
    ┌───▼─────────▼──┐  ┌────▼────┐  ┌────▼────┐
    │  Scorer        │  │Embedder │  │Summarizer
    │  (GPT-2)       │  │(MiniLM) │  │(DistilBART)
    │  - Logprobs    │  │- Sim    │  │- T5 cleanup
    │  - Surprisal   │  │- Encode │  │- Summarize
    └────────────────┘  └─────────┘  └─────────┘
```

---

## 📦 Installation

### Prerequisites
- **Python 3.10+**
- **pip** or **poetry** (recommended)
- **CUDA** (optional, for GPU acceleration)

### Setup

#### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/ranveer0323/prompt-compressor.git
cd prompt-compressor

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/ranveer0323/prompt-compressor.git
cd prompt-compressor

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Optional: Gemini API key for LLM generation
GOOGLE_API_KEY=your_api_key_here

# Optional: Flask configuration
FLASK_ENV=development
FLASK_DEBUG=True
```

Or set environment variables directly:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

---

## 🚀 Quick Start

### 1. Start the Backend

```bash
python app.py
```

The Flask server will start on `http://localhost:7860`

### 2. Open the Web UI

Navigate to `http://localhost:7860` in your browser.

### 3. Compress a Prompt

1. **Paste your prompt** in the "Prompt Input" textarea
2. **Adjust settings** (optional):
   - Keep Ratio: How much of the original to retain (0.4–1.0)
   - Max Phrase Length: Maximum words per removable phrase (2–4)
   - Similarity Threshold: Minimum semantic similarity to preserve (0.5–0.99)
3. **Click "Analyze"** to see phrase candidates and surprisal distribution
4. **Click "Phrase Prune"** or **"Hybrid Compress"** to compress
5. **Click "Generate (Gemini)"** to see LLM outputs (requires API key)

### Example Workflow

```
Original Prompt (500 tokens)
    ↓
[Click "Analyze"]
    ↓
See phrase candidates sorted by surprisal
    ↓
[Click "Phrase Prune"]
    ↓
Compressed Prompt (425 tokens, 85% kept)
    ↓
[Click "Generate (Gemini)"]
    ↓
Compare outputs: Original vs. Pruned
```

---

## 🔌 API Endpoints

All endpoints accept JSON POST requests and return JSON responses.

### `/api/analyze`

Analyzes a prompt and returns phrase candidates and surprisal scores.

**Request:**
```json
{
  "prompt": "Your prompt text here..."
}
```

**Response:**
```json
{
  "words": ["Your", "prompt", "text", ...],
  "word_spans": [[0, 1], [1, 2], ...],
  "phrase_candidates_count": 150,
  "top_phrase_candidates": [
    {"phrase": "Your prompt", "norm": 0.12},
    {"phrase": "prompt text", "norm": 0.15},
    ...
  ],
  "token_count": 42
}
```

---

### `/api/prune`

Runs greedy phrase-level pruning compression.

**Request:**
```json
{
  "prompt": "Your prompt text here...",
  "keep_ratio": 0.85,
  "max_phrase_len": 4,
  "sim_threshold": 0.82
}
```

**Response:**
```json
{
  "run_id": "uuid-string",
  "compressed": "Compressed prompt text...",
  "sim": 0.89,
  "log_len": 5
}
```

---

### `/api/hybrid`

Runs hybrid compression (rule-based + summarization + pruning).

**Request:**
```json
{
  "prompt": "Your prompt text here...",
  "sim_threshold": 0.82,
  "keep_ratio_phrases": 0.6
}
```

**Response:**
```json
{
  "run_id": "uuid-string",
  "hybrid": "Hybrid compressed prompt text..."
}
```

---

### `/api/generate`

Generates LLM output using Gemini API.

**Request (using run_id):**
```json
{
  "which": "original",
  "run_id": "uuid-string",
  "model": "gemini-2.5-flash"
}
```

**Request (inline prompt):**
```json
{
  "prompt": "Your prompt text here...",
  "model": "gemini-2.5-flash"
}
```

**Response:**
```json
{
  "text": "LLM generated output..."
}
```

---

### `/api/validate`

Computes semantic similarity between two texts.

**Request:**
```json
{
  "a": "First text...",
  "b": "Second text..."
}
```

**Response:**
```json
{
  "similarity": 0.87,
  "tokens_a": 42,
  "tokens_b": 38
}
```

---

## 🧠 Algorithms

### Algorithm 1: Phrase-Level Surprisal-Based Greedy Pruning

#### What is Surprisal?

Surprisal is an information-theoretic measure of how "surprising" (unpredictable) a token is:

```
Surprisal(token) = -log P(token | previous context)
```

- **Low surprisal** = predictable, redundant, low-information
- **High surprisal** = meaningful, content-bearing, important

#### Steps

1. **Tokenization**: Split prompt into words using regex-based tokenizer
2. **GPT-2 Scoring**: Get per-token log-probabilities from GPT-2
3. **Word Surprisal**: Aggregate token surprisals per word
4. **Phrase Candidates**: Generate 1–4 word phrases with average surprisal
5. **Protection**: Mark structural elements (labels, examples, stopwords) as non-removable
6. **Greedy Removal**: Iteratively remove lowest-surprisal phrase, recompute surprisals
7. **Semantic Validation**: Stop if cosine similarity drops below threshold
8. **Output**: Compressed prompt with iteration log

#### Example

```
Original: "The quick brown fox jumps over the lazy dog"

Surprisals:
  "The" → 0.05 (low, common)
  "quick" → 0.45 (medium)
  "brown" → 0.42 (medium)
  "fox" → 0.60 (high, specific)
  ...

Candidates (sorted by surprisal):
  1. "The" (0.05) ← Remove first
  2. "the lazy" (0.12)
  3. "quick brown" (0.43)
  ...

After removal: "Quick brown fox jumps over lazy dog"
```

---

### Algorithm 2: Hybrid Compression Pipeline

#### Steps

1. **Rule-Based Compression**: Simplify structure (bullets, repeated adjectives)
2. **Section Splitting**: Split by headers (## Writing Guidelines, etc.)
3. **Selective Summarization**: Summarize long sections using DistilBART
4. **Similarity Check**: If summary similarity < threshold, fall back to pruning
5. **Fallback Pruning**: Use greedy phrase pruning if summarization fails
6. **Assembly**: Combine all sections into final compressed prompt

#### Example

```
Original (500 tokens):
  ## Writing Guidelines:
    [Long paragraph about tone, style, formatting...]
  ## Examples:
    [Multiple examples...]

After Hybrid:
  ## Writing Guidelines:
    [Summarized to 50 tokens]
  ## Examples:
    [Pruned to 80 tokens]

Result: 200 tokens (60% reduction)
```

---

## 📁 Project Structure

```
prompt-compressor/
├── app.py                      # Flask backend, routes
├── pyproject.toml              # Poetry dependencies
├── .env                        # Environment variables (API keys)
├── .gitignore                  # Git ignore rules
│
├── compressor/                 # Compression algorithms
│   ├── __init__.py
│   ├── pruner.py              # Greedy phrase pruning
│   ├── hybrid.py              # Hybrid compression pipeline
│   └── utils.py               # Helper functions (rule compression, cleanup)
│
├── models/                     # ML models
│   ├── __init__.py
│   ├── scorer.py              # GPT-2 scorer (surprisal)
│   ├── embedder.py            # SentenceTransformer (similarity)
│   └── summarizer.py          # DistilBART + T5 (summarization)
│
├── templates/                  # Frontend
│   └── index.html             # Main UI
│
├── static/                     # Frontend assets
│   ├── app.js                 # JavaScript logic
│   └── style.css              # Modern styling
│
└── README.md                   # This file
```

---

## ⚙️ Configuration

### Flask Configuration

Create `instance/config.py` for persistent configuration:

```python
# instance/config.py
GOOGLE_API_KEY = "your_api_key_here"
DEBUG = True
TESTING = False
```

### Model Configuration

Models are loaded as singletons in `app.py`:

```python
SCORER = Scorer()           # GPT-2 (for surprisal)
EMBEDDER = Embedder()       # SentenceTransformer MiniLM (for similarity)
SUMMARIZER = Summarizer()   # DistilBART (for summarization)
```

To customize models, edit `models/scorer.py`, `models/embedder.py`, or `models/summarizer.py`.

### Compression Parameters

Adjust defaults in the UI or API:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `keep_ratio` | 0.85 | 0.4–1.0 | Fraction of words to keep |
| `max_phrase_len` | 4 | 2–4 | Max words per removable phrase |
| `sim_threshold` | 0.82 | 0.5–0.99 | Min semantic similarity |

---

## 🐛 Troubleshooting

### Issue: "GOOGLE_API_KEY missing"

**Solution**: Set the API key in `.env` or environment:

```bash
export GOOGLE_API_KEY="your_key_here"
python app.py
```

### Issue: Models take too long to load

**Solution**: Models are cached after first load. Subsequent requests are fast. First startup may take 1–2 minutes.

### Issue: CUDA out of memory

**Solution**: Use CPU mode or reduce batch sizes. Edit `models/scorer.py`:

```python
self.device = "cpu"  # Force CPU
```

### Issue: Similarity always returns -

**Solution**: Ensure both texts are non-empty. Check `/api/validate` endpoint directly.

### Issue: Frontend not loading

**Solution**: Ensure Flask is running on `localhost:7860`. Check browser console for errors.

---

## 🚀 Future Enhancements

### Phase 1: UI Improvements
- [ ] Diff view highlighting removed phrases
- [ ] Per-word surprisal tooltip on hover
- [ ] Real-time compression progress bar
- [ ] Dark mode toggle
- [ ] Export as PDF report

### Phase 2: Advanced Algorithms
- [ ] Semantic clustering for phrase grouping
- [ ] Multi-language support
- [ ] Custom tokenizers
- [ ] Fine-tuned surprisal models

### Phase 3: Deployment
- [ ] Docker containerization
- [ ] Docker Compose with GPU support
- [ ] Kubernetes manifests
- [ ] Cloud deployment (AWS, GCP, Azure)

### Phase 4: Integration
- [ ] LangChain integration
- [ ] OpenAI API support
- [ ] Anthropic Claude support
- [ ] Ollama local model support

---

## 📊 Performance Benchmarks

Tested on a 2000-token prompt with default settings:

| Metric | Value |
|--------|-------|
| Analysis time | ~2s |
| Phrase pruning time | ~5s |
| Hybrid compression time | ~8s |
| Gemini generation time | ~10s |
| **Total end-to-end** | ~25s |

---

## 📝 Example Use Cases

### 1. Cost Optimization
Compress system prompts for production LLM APIs to reduce per-token costs.

### 2. Latency Reduction
Shorten prompts to speed up inference on edge devices or rate-limited APIs.

### 3. Token Limit Compliance
Fit long prompts within model token limits (e.g., Claude's 200k context).

### 4. Prompt Engineering
Analyze which phrases are redundant and refactor prompts for clarity.

### 5. Research & Analysis
Study how compression affects LLM behavior and output quality.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add your feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

## 👤 Author

**Ranveer Singh**
- GitHub: [@ranveer0323](https://github.com/ranveer0323)
- Email: ranawatranveer0323@gmail.com

---

## 🙏 Acknowledgments

- **GPT-2** (OpenAI) for token logprobs
- **SentenceTransformer** (Hugging Face) for embeddings
- **DistilBART** (Hugging Face) for summarization
- **Gemini API** (Google) for LLM generation
- **Flask** for the web framework
- **Bootstrap 5** for UI components

---

## 📞 Support

For issues, questions, or suggestions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Open an [Issue](https://github.com/ranveer0323/prompt-compressor/issues)
3. Start a [Discussion](https://github.com/ranveer0323/prompt-compressor/discussions)

---

**Last Updated**: November 2025

**Status**: ✅ Fully Functional | 🚀 Production Ready
