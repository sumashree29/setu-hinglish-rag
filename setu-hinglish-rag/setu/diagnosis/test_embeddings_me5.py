"""
Sanity test for multilingual-e5-large embedding model.

NOTE: e5 models expect a specific prefix convention: "query: " for queries
and "passage: " for documents/passages. This isn't optional -- e5 models are
trained with these prefixes and perform noticeably worse without them.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-large"

print(f"Loading {MODEL_NAME} ... (this downloads ~2.2GB on first run, be patient)")
model = SentenceTransformer(MODEL_NAME)

sample_query = "query: BSBDA account mein monthly kitne free withdrawals milte hain"
sample_doc = "passage: Banks must provide at least four free cash withdrawals per month on a BSBDA."

emb_q = model.encode(sample_query)
emb_d = model.encode(sample_doc)

print("Query embedding shape:", emb_q.shape)
print("Doc embedding shape:", emb_d.shape)

cos_sim = np.dot(emb_q, emb_d) / (np.linalg.norm(emb_q) * np.linalg.norm(emb_d))
print("Cosine similarity (query vs relevant doc):", cos_sim)