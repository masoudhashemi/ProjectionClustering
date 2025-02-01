from sentence_transformers import SentenceTransformer


class EmbeddingProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize the SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encode a list of texts into embeddings.
        """
        return self.model.encode(texts)
