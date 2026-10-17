import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Transforme une liste de textes en vecteurs denses unitaires (dim: 384)."""
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=show_progress,
            normalize_embeddings=True
        )