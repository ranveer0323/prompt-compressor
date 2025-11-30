import re
from compressor.utils import cleanup_text_t5
from models.scorer import Scorer
from models.embedder import Embedder
from sentence_transformers import util
import torch

# Common English stop words
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
        if hasattr(scorer_or_tokenizer, "encode_no_special"):
            toks = scorer_or_tokenizer.encode_no_special(w)
        else:
            toks = scorer_or_tokenizer.tokenizer.encode(w, add_special_tokens=False)
        spans.append((len(flattened), len(flattened) + len(toks)))
        flattened.extend(toks)
    return flattened, spans

def phrase_surprisals_for_segment(segment_text, scorer=None, max_phrase_len=4):
    scorer = scorer or Scorer()
    words = simple_word_tokenize(segment_text)
    flat_ids, spans = words_to_token_ids(words, scorer)
    token_logprobs = scorer.token_logprobs_for_sequence(flat_ids)
    
    n = len(words)
    label_pattern = re.compile(r'^(Input:|Headline:|Tagline:|Output:|Example|##\s)', re.I)
    protected = set()
    
    for idx, w in enumerate(words):
        # Protect quotes
        if (w.startswith('"') and w.endswith('"')) or (w.startswith("'") and w.endswith("'")):
            protected.add(idx)
        # Protect labels
        if label_pattern.match(w):
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
            
            s_tok = spans[i][0]
            e_tok = spans[j][1]
            surr_sum = -sum(token_logprobs[s_tok:e_tok]) if (e_tok > s_tok) else 1e6
            tok_count = max(1, e_tok - s_tok)
            norm = surr_sum / tok_count
            
            candidates.append({
                'i': i, 
                'j': j, 
                'norm': norm, 
                'phrase': " ".join(words[i:j+1]),
                'indices': set(range(i, j+1))
            })
            
    candidates = sorted(candidates, key=lambda x: x['norm'])
    return words, spans, candidates, protected

def greedy_phrase_prune(segment_text, keep_ratio=0.85, max_phrase_len=4, similarity_threshold=0.82, scorer=None, embedder=None, max_removals=200):
    scorer = scorer or Scorer()
    embedder = embedder or Embedder()
    original = segment_text
    
    # Initial Calculation
    words = simple_word_tokenize(segment_text)
    target_len = max(1, int(len(words) * keep_ratio))
    curr_text = segment_text
    curr_words = words[:]
    
    iteration_log = []
    total_removed_count = 0
    
    # BATCH SIZE: How many phrases to remove before re-running GPT-2
    # Increasing this speeds up the code significantly but might be slightly less precise
    BATCH_SIZE = 10 

    while len(curr_words) > target_len and total_removed_count < max_removals:
        # 1. Run Scorer (The slow part)
        words_now, spans, phrase_scores, protected = phrase_surprisals_for_segment(curr_text, scorer=scorer, max_phrase_len=max_phrase_len)
        
        # 2. Identify a batch of non-overlapping phrases to remove
        indices_to_remove = set()
        batch_removed = 0
        
        for ph in phrase_scores:
            if batch_removed >= BATCH_SIZE:
                break
            
            # Check overlap with already marked indices in this batch
            if not ph['indices'].isdisjoint(indices_to_remove):
                continue
                
            # Safety Checks on the phrase string
            phrase_str = ph['phrase']
            if any(re.match(r'^[\:\'\"\-\—\(\)\[\]\{\}]$', w) for w in phrase_str.split()):
                continue
            
            # If removing this creates unbalanced quotes, skip (simple heuristic)
            # (Skipping complex quote balancing for speed here)
            
            # Mark for removal
            indices_to_remove.update(ph['indices'])
            
            # Log it
            total_removed_count += 1
            batch_removed += 1
            iteration_log.append({
                'iter': total_removed_count, 
                'removed_phrase': phrase_str, 
                'norm': ph['norm'], 
                'current_word_count': len(curr_words) - len(indices_to_remove) # Approx
            })

        # 3. If nothing found to remove, stop
        if batch_removed == 0:
            break
            
        # 4. Reconstruct text
        new_word_list = []
        for idx, w in enumerate(words_now):
            if idx not in indices_to_remove:
                new_word_list.append(w)
        
        curr_words = new_word_list
        curr_text = " ".join(curr_words)
        curr_text = re.sub(r'\s+([.,:;?!])', r'\1', curr_text) # Fix punctuation spacing
        
        # Check lengths
        if len(curr_words) <= target_len:
            break

    compressed = curr_text.strip()
    
    # Semantic validation
    emb_o = embedder.encode(original, convert_to_tensor=True)
    emb_c = embedder.encode(compressed, convert_to_tensor=True)
    sim = float(util.pytorch_cos_sim(emb_o, emb_c).item())
    
    return compressed, sim, iteration_log