import re
from compressor.utils import rule_compress
from models.summarizer import Summarizer
from models.embedder import Embedder
from compressor.pruner import greedy_phrase_prune

def hybrid_compress(full_prompt, summarizer=None, embedder=None, scorer=None, sim_thresh=0.82, keep_ratio_phrases=0.6):
    summarizer = summarizer or Summarizer()
    embedder = embedder or Embedder()
    scorer = scorer  # used by fallback pruner if needed
    # 1) structural pass
    after_rule = rule_compress(full_prompt)

    # 2) split by headers
    chunks = re.split(r'(\n## [^\n]+)', after_rule)
    assembled = []
    i = 0
    while i < len(chunks):
        if chunks[i].strip() == "":
            i += 1
            continue
        if chunks[i].startswith("\n## ") or chunks[i].startswith("## "):
            header = chunks[i].strip()
            body = chunks[i+1] if (i+1) < len(chunks) else ""
            # choose to summarize for certain headers
            if header.strip() in ["## Writing Guidelines:", "## Example:", "## Style Examples by Brand Type:"]:
                try:
                    summ = summarizer.summarize(body, max_length=60, min_length=10)
                    sim = embedder.similarity(body, summ)
                    if sim < sim_thresh:
                        # fallback to phrase prune
                        pruned, sim2, log = greedy_phrase_prune(body, keep_ratio=keep_ratio_phrases, scorer=scorer, embedder=embedder)
                        if sim2 < sim_thresh:
                            assembled.append(header + "\n" + body)
                        else:
                            assembled.append(header + "\n" + pruned)
                    else:
                        assembled.append(header + "\n" + summ)
                except Exception:
                    pruned, sim2, log = greedy_phrase_prune(body, keep_ratio=keep_ratio_phrases, scorer=scorer, embedder=embedder)
                    if sim2 < sim_thresh:
                        assembled.append(header + "\n" + body)
                    else:
                        assembled.append(header + "\n" + pruned)
            else:
                assembled.append(header + "\n" + body)
            i += 2
        else:
            assembled.append(chunks[i])
            i += 1

    final = "\n".join(assembled)
    final = re.sub(r'\n\s+\n', '\n\n', final).strip()
    return final
