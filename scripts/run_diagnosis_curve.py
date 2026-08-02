"""
Phase 1 exit deliverable: CMI-vs-Recall/MRR/nDCG curve for all 3 embedding models.
OWNER: R1 | PHASE: 1 (plan §3.6)
Run after build_pilot_corpus.py.
"""
import matplotlib.pyplot as plt
from setu.config import EMBEDDING_MODELS, RESULTS_FIGS

def main():
    """
    TODO (R1):
        1. Load pilot corpus + queries
        2. For each embedding model: embed corpus, build FAISS index,
           retrieve top-k for each query
        3. Compute Recall@k / MRR / nDCG@5 per query (setu.evaluation.metrics)
        4. Bin queries by CMI (config.CMI_BANDS), average metrics per bin
        5. Plot each metric vs CMI bin, one line per embedding model
        6. Save to results/figures/diagnosis_curve.png
    """
    raise NotImplementedError("R1: implement end-to-end diagnosis curve script")

if __name__ == "__main__":
    main()
