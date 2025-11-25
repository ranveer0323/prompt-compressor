from sentence_transformers import SentenceTransformer, util
import torch

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text, convert_to_tensor=False):
        return self.model.encode(text, convert_to_tensor=convert_to_tensor)

    def similarity(self, a, b):
        ea = self.encode(a, convert_to_tensor=True)
        eb = self.encode(b, convert_to_tensor=True)
        return float(util.pytorch_cos_sim(ea, eb).item())
