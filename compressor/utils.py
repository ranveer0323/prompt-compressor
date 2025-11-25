import re
from transformers import pipeline
import torch

# optional grammar cleanup using t5-small
_cleaner = None
def cleanup_text_t5(text, max_len=256):
    global _cleaner
    if _cleaner is None:
        device = 0 if torch.cuda.is_available() else -1
        _cleaner = pipeline("text2text-generation", model="t5-small", device=device)
    prompt = "Fix grammar and punctuation while preserving meaning and concision:\n\n" + text
    out = _cleaner(prompt, max_length=max_len, do_sample=False)
    cleaned = out[0]['generated_text']
    cleaned = re.sub(r'\s+([.,:;?!])', r'\1', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
    return cleaned

def rule_compress(text):
    t = text
    t = re.sub(r'\b(emotionally engaging|brand-aligned|on-brand|authenticity|deeply)\b', '', t, flags=re.I)
    t = re.sub(r'## Objectives:\n(?:[0-9\.\-\s\w\,\']+\n)+', '## Objectives: Understand brand; Keep tone; Benefits-first; Clear CTA\n', t)
    t = re.sub(r'## Writing Guidelines:\n((?:- .*\n)+)', '## Writing Guidelines: be concise, vivid, avoid passive voice, benefit-first, strong CTA\n', t)
    t = re.sub(r'## Output Formats:\n((?:- .*\n)+)', '## Output Formats: headlines, emails, landing pages, ads, scripts\n', t)
    # compact examples into one-liners
    def example_repl(m):
        body = m.group(0)
        inp = re.search(r'Input:\s*(["\'].*?["\']|.+)', body)
        out = re.search(r'Output:\s*([\s\S]+)', body)
        short = "## Example: " + (inp.group(1) if inp else "Example") + " -> "
        if out:
            first_line = out.group(1).strip().splitlines()[0]
            short += first_line
        return short + "\n"
    t = re.sub(r'## Example:[\s\S]*?(?=\n## |$)', example_repl, t)
    t = re.sub(r'\n\s+\n', '\n\n', t)
    t = re.sub(r'\s{2,}', ' ', t)
    return t.strip()
