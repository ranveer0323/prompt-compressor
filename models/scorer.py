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
        """
        Return list of log probs.
        For a sequence [A, B, C], we want:
        - log P(B|A)
        - log P(C|A,B)
        (The first token A has no conditional probability in this context, 
         so we usually assign it 0 or ignore it. Here we return 0.0 for the first token).
        """
        if len(token_ids) < 2:
            return [0.0] * len(token_ids)
            
        ids = torch.tensor([token_ids], device=self.device)
        outputs = self.model(ids)
        
        # Logits shape: (1, seq_len, vocab_size)
        # Shift logits and labels:
        # logits[0, :-1] predicts the next token
        # labels[0, 1:] are the actual next tokens
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = ids[0, 1:]
        
        # Gather log probabilities
        # log_softmax gives log probabilities across vocab
        log_probs_all = torch.log_softmax(shift_logits, dim=-1)
        
        # Select the log prob of the actual token that appeared
        # gather expects index to have same dimensions, so we unsqueeze
        target_log_probs = log_probs_all.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        
        # Result list: First token gets 0.0 (or high probability) as it's the start context
        # Convert tensor to list of floats
        return [0.0] + target_log_probs.tolist()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode_no_special(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)
