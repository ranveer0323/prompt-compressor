import re
from compressor.utils import cleanup_text_t5
from models.scorer import Scorer
from models.embedder import Embedder
from sentence_transformers import util
import torch

# NOTE: this function expects scorer (Scorer instance) and embedder (Embedder instance) or will construct defaults locally.
STOP_PROTECT = {
    "a","an","the","in","on","for","with","by","to","and","or","but","of","at","as","from","into","over","per"
}

def simple_word_tokenize(text):
    parts = re.findall(r'\"[^\"]*\"|\'[^\']*\'|\w+|[^\s\w]', text)
    return parts

def words_to_token_ids(words, scorer_or_tokenizer):
    flattened = []
    spans = []
    for w in words:
        toks = scorer_or_tokenizer.encode_no_special(w) if hasattr(scorer_or_tokenizer, "encode_no_special") else scorer_or_tokenizer.tokenizer.encode(w, add_special_tokens=False)
        spans.append((len(flattened), len(flattened) + len(toks)))
        flattened.extend(toks)
    return flattened, spans

def phrase_surprisals_for_segment(segment_text, scorer=None, max_phrase_len=4):
    scorer = scorer or Scorer()
    words = simple_word_tokenize(segment_text)
    flat_ids, spans = words_to_token_ids(words, scorer)
    token_logprobs = scorer.token_logprobs_for_sequence(flat_ids)
    per_word = []
    for (s, e) in spans:
        surr = -sum(token_logprobs[s:e]) if e > s else 1e6
        norm = surr / max(1, (e - s))
        per_word.append({'s': s, 'e': e, 'surr': surr, 'norm': norm})
    # build phrase candidates
    n = len(words)
    label_pattern = re.compile(r'^(Input:|Headline:|Tagline:|Output:|Example|##\s)', re.I)
    protected = set()
    for idx, w in enumerate(words):
        if (w.startswith('"') and w.endswith('"')) or (w.startswith("'") and w.endswith("'")):
            protected.add(idx)
        if label_pattern.match(w):
            protected.add(idx)
        if w.lower() in STOP_PROTECT:
            protected.add(idx)
    candidates = []
    for i in range(n):
        if i in protected:
            continue
        for L in range(1, max_phrase_len + 1):
            j = i + L - 1
            if j >= n:
                break
            if any(k in protected for k in range(i, j+1)):
                break
            # require multi-word unless content word
            if L == 1 and words[i].lower() in STOP_PROTECT:
                continue
            s_tok = spans[i][0]
            e_tok = spans[j][1]
            surr_sum = -sum(token_logprobs[s_tok:e_tok]) if (e_tok> s_tok) else 1e6
            tok_count = max(1, e_tok - s_tok)
            norm = surr_sum / tok_count
            candidates.append({'i': i, 'j': j, 's_tok': s_tok, 'e_tok': e_tok, 'surr': surr_sum, 'norm': norm, 'phrase': " ".join(words[i:j+1])})
    # sort ascending (low norm = most predictable)
    candidates = sorted(candidates, key=lambda x: x['norm'])
    return words, spans, candidates, protected

def greedy_phrase_prune(segment_text, keep_ratio=0.85, max_phrase_len=4, similarity_threshold=0.82, scorer=None, embedder=None, max_removals=200):
    scorer = scorer or Scorer()
    embedder = embedder or Embedder()
    original = segment_text
    words = simple_word_tokenize(segment_text)
    target_len = max(1, int(len(words) * keep_ratio))
    curr_words = words[:]
    iteration_log = []
    removals = 0

    while len(curr_words) > target_len and removals < max_removals:
        joined = " ".join(curr_words)
        words_now, spans, phrase_scores, protected = phrase_surprisals_for_segment(joined, scorer=scorer, max_phrase_len=max_phrase_len)
        removed = False
        for ph in phrase_scores:
            i, j = ph['i'], ph['j']
            phrase_words = words_now[i:j+1]
            # safety checks
            if any(re.match(r'^[\:\'\"\-\—\(\)\[\]\{\}]$', w) for w in phrase_words):
                continue
            new_words = words_now[:i] + words_now[j+1:]
            new_text = " ".join(new_words)
            new_text = re.sub(r'\s+([.,:;?!])', r'\1', new_text)
            if new_text.count('"') % 2 != 0 or new_text.count("'") % 2 != 0:
                continue
            # accept removal
            curr_words = new_words
            removals += 1
            iteration_log.append({'iter': removals, 'removed_phrase': ph['phrase'], 'i': i, 'j': j, 'norm': ph['norm'], 'current_word_count': len(curr_words)})
            removed = True
            break
        if not removed:
            break

    compressed = " ".join(curr_words)
    compressed = re.sub(r'\s+([.,:;?!])', r'\1', compressed).strip()
    # optional cleanup
    cleaned = cleanup_text_t5(compressed)
    # semantic validation
    emb_o = embedder.encode(original, convert_to_tensor=True)
    emb_c = embedder.encode(cleaned, convert_to_tensor=True)
    sim = float(util.pytorch_cos_sim(emb_o, emb_c).item())
    if sim < similarity_threshold:
        # fallback to original or a safer structural compress
        return original, 1.0, iteration_log
    return cleaned, sim, iteration_log
