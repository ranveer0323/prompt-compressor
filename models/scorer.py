import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import math

class Scorer:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = self.model.config.n_positions  # Usually 1024

    @torch.no_grad()
    def token_logprobs_for_sequence(self, token_ids):
        """
        Calculates logprobs using a SLIDING WINDOW (Stride).
        Ensures context is preserved across the 1024-token boundary.
        """
        if not token_ids:
            return []

        seq_len = len(token_ids)
        
        # If short enough, run normally
        if seq_len <= self.max_length:
            return self._compute_logprobs(token_ids)

        # STRIDE STRATEGY: Process in windows of 1024, overlapping by 512
        stride = 512
        logprobs_out = [0.0] * seq_len
        
        # 1. Process the first window fully
        first_window_ids = token_ids[:self.max_length]
        first_scores = self._compute_logprobs(first_window_ids)
        for i in range(len(first_scores)):
            logprobs_out[i] = first_scores[i]
            
        # 2. Process subsequent windows
        current_pos = stride
        while current_pos < seq_len:
            end_pos = min(current_pos + stride, seq_len)
            window_start = max(0, end_pos - self.max_length)
            window_ids = token_ids[window_start:end_pos]
            
            window_scores = self._compute_logprobs(window_ids)
            
            # Update only the new part (tail of the window)
            update_len = end_pos - current_pos
            relevant_scores = window_scores[-update_len:]
            
            for i in range(update_len):
                logprobs_out[current_pos + i] = relevant_scores[i]
            
            current_pos += stride

        return logprobs_out

    def _compute_logprobs(self, token_ids):
        """Helper to run model on a valid batch size"""
        if len(token_ids) < 2:
            return [0.0] * len(token_ids)

        ids = torch.tensor([token_ids], device=self.device)
        outputs = self.model(ids)
        
        # Shift logits (prediction of next token)
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = ids[0, 1:]
        
        log_probs_all = torch.log_softmax(shift_logits, dim=-1)
        target_log_probs = log_probs_all.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
        
        return [0.0] + target_log_probs.tolist()

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode_no_special(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)