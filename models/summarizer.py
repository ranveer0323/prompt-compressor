from transformers import pipeline
import torch

class Summarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("summarization", model=model_name, device=device)

    def summarize(self, text, max_length=80, min_length=10):
        out = self.pipe(text, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]['summary_text']
