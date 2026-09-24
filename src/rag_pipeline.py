import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

class RAGIndex:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.model = SentenceTransformer(MODEL_NAME)
        texts = (self.df["question"].fillna("") + " " + self.df["answer"]).tolist()
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings.astype(np.float32))

    def search(self, query, top_k=6):
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb.astype(np.float32), top_k)
        results = self.df.iloc[idxs[0]].copy()
        results["score"] = scores[0]
        return results