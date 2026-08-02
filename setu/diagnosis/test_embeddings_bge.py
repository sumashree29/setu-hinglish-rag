"""
Sanity test for BGE-M3 embedding model.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "BAAI/bge-m3"

print(f"Loading {MODEL_NAME} ... (this downloads ~2.2GB on first run, be patient)")
model = SentenceTransformer(MODEL_NAME)

sample_query = "BSBDA account mein monthly kitne free withdrawals milte hain"
sample_doc = "Banks must provide at least four free cash withdrawals per month on a BSBDA."

emb_q = model.encode(sample_query)
emb_d = model.encode(sample_doc)

print("Query embedding shape:", emb_q.shape)
print("Doc embedding shape:", emb_d.shape)

cos_sim = np.dot(emb_q, emb_d) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_d))
print("Cosine similarity (query vs relevant doc):", cos_sim)