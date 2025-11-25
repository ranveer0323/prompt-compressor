import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import math

class Scorer:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        # ensure padding token exists for tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def token_logprobs_for_sequence(self, token_ids):
        """Return list of log probs (log p(token_i | tokens[:i])) for each token in token_ids"""
        if len(token_ids) == 0:
            return []
        ids = torch.tensor([token_ids], device=self.device)
        outputs = self.model(ids)
        logits = outputs.logits.squeeze(0)  # (seq_len, vocab)
        logprobs = torch.log_softmax(logits, dim=-1)
        # For token at position i, take logprob at position i for token token_ids[i]
        out = [float(logprobs[pos, tok].item()) for pos, tok in enumerate(token_ids)]
        return out

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode_no_special(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
