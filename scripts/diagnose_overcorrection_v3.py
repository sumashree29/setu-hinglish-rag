import sys
import json
import numpy as np
import faiss
import pickle
import time
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from sentence_transformers import SentenceTransformer

from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.operators.caep import extract_entity_list, entity_frequencies, apply_caep
from setu.operators.lqp import apply_lqp
from setu.operators.lag import fit_lag_v2, predict_strategy, apply_lag, entity_density
from setu.fusion.carf import rrf_baseline
from setu.controller.setu_bandit import setu_v1_fixed_order

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "results" / "models"
LOGS_DIR = ROOT / "results" / "logs"

# 1. Load Data
print("Loading data...")
corpus_lines = [json.loads(l) for l in open(DATA_DIR / "processed" / "corpus_chunks_v2.jsonl", encoding="utf-8")]
queries = json.load(open(DATA_DIR / "processed" / "queries_v3_final.json", encoding="utf-8"))
lag_labeled_data = json.load(open(DATA_DIR / "processed" / "lag_labels_v3.json", encoding="utf-8"))
v2_logs = json.load(open(LOGS_DIR / "setu_v2_per_query_v3.json", encoding="utf-8"))
v2_ranking_dict = {q["query_id"]: q["final_ranking"] for q in v2_logs}

doc_ids = [c["chunk_id"] for c in corpus_lines]
doc_texts = [c["text"] for c in corpus_lines]
query_ids = [q["query_id"] for q in queries]
q_by_id = {q["query_id"]: q for q in queries}

query_cmi = {q["query_id"]: cmi(q["text"]) for q in queries}
query_entropy = {q["query_id"]: lid_entropy(q["text"]) for q in queries}

entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)

qrels = {q["query_id"]: set(q["relevant_doc_ids"]) for q in queries}

def get_mrr(ranking, rel_docs):
    for r, did in enumerate(ranking):
        if did in rel_docs:
            return 1.0 / (r + 1)
    return 0.0

EMBEDDING_MODELS = {
    "bge_m3": ("BAAI/bge-m3", False),
    "indic_sbert": ("l3cube-pune/indic-sentence-similarity-sbert", False),
    "me5_large": ("intfloat/multilingual-e5-large", True)
}

final_results = {}
plot_data = {}

for model_key, (model_name, needs_prefix) in EMBEDDING_MODELS.items():
    print(f"\n==========================================")
    print(f"Processing Model: {model_key}")
    print(f"==========================================")
    
    st_model = SentenceTransformer(model_name)
    def embed_fn(texts):
        inp = [f"query: {t}" for t in texts] if needs_prefix else texts
        return st_model.encode(inp, convert_to_numpy=True).astype("float32")

    doc_emb = np.load(LOGS_DIR / f"doc_emb_{model_key}_v2.npy")
    query_emb_raw = np.load(LOGS_DIR / f"query_emb_{model_key}_v3.npy")
    
    faiss.normalize_L2(doc_emb)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    def faiss_search(q_emb, k=10):
        q = np.asarray(q_emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, k)
        return [doc_ids[i] for i in indices[0]]

    # Load operators
    lqp_model = pickle.load(open(MODELS_DIR / f"lqp_model_{model_key}.pkl", "rb"))
    caep_gate = pickle.load(open(MODELS_DIR / f"caep_gate_{model_key}.pkl", "rb"))
    
    # 5-fold splits for LAG (reproducing Phase 10 exact logic)
    n_splits = 5
    folds = np.array_split(query_ids, n_splits)
    lag_models_per_fold = {}
    for fold_idx, test_qids in enumerate(folds):
        train_ids = set(query_ids) - set(test_qids)
        fold_lag_data = [d for d in lag_labeled_data if d["query_id"] in train_ids]
        X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in fold_lag_data])
        y_lag = np.array([d["label"] for d in fold_lag_data])
        lag_models_per_fold[fold_idx] = fit_lag_v2(X_lag, y_lag)
        
    def get_lag_model_for_qid(qid):
        for fold_idx, test_qids in enumerate(folds):
            if qid in test_qids:
                return lag_models_per_fold[fold_idx]
        return None

    # Tracking deltas
    deltas = {
        "LQP": {"correct": [], "incorrect": []},
        "CAEP": {"correct": [], "incorrect": []},
        "LAG": {"correct": [], "incorrect": []},
    }
    if model_key == "bge_m3":
        deltas["SETU_v1"] = {"correct": [], "incorrect": []}
        deltas["SETU_v2"] = {"correct": [], "incorrect": []}

    raw_correct_count = 0
    raw_incorrect_count = 0
    
    for i, q in enumerate(queries):
        qid = q["query_id"]
        q_text = q["text"]
        rel = qrels[qid]
        
        # RAW
        raw_e = query_emb_raw[i]
        r_rank = faiss_search(raw_e)
        raw_mrr = get_mrr(r_rank, rel)
        
        if raw_mrr == 1.0:
            group = "correct"
            raw_correct_count += 1
        else:
            group = "incorrect"
            raw_incorrect_count += 1
            
        # LQP
        l_e = apply_lqp(raw_e, query_cmi[qid], lqp_model, cmi_max=1.0)
        l_rank = faiss_search(l_e)
        deltas["LQP"][group].append(get_mrr(l_rank, rel) - raw_mrr)
        
        # CAEP
        c_text = apply_caep(q_text, entities, caep_gate, entity_freq, embed_fn=embed_fn)
        if c_text != q_text:
            c_e = embed_fn([c_text])[0]
            c_rank = faiss_search(c_e)
        else:
            c_rank = r_rank
        deltas["CAEP"][group].append(get_mrr(c_rank, rel) - raw_mrr)
        
        # LAG
        lag_model = get_lag_model_for_qid(qid)
        q_dens = entity_density(q_text, entities)
        lag_strat = predict_strategy(query_cmi[qid], query_entropy[qid], q_dens, lag_model)
        lag_out = apply_lag(q_text, lag_strat, entities, embed_fn, caep_gate, entity_freq)
        
        if isinstance(lag_out, list):
            r1 = faiss_search(embed_fn([lag_out[0]])[0])
            r2 = faiss_search(embed_fn([lag_out[1]])[0])
            lag_rank = rrf_baseline([r1, r2])
        else:
            lag_rank = faiss_search(embed_fn([lag_out])[0])
        deltas["LAG"][group].append(get_mrr(lag_rank, rel) - raw_mrr)
        
        # SETU v1 and v2 (only for BGE-M3)
        if model_key == "bge_m3":
            # v2
            v2_rank = v2_ranking_dict[qid]
            deltas["SETU_v2"][group].append(get_mrr(v2_rank, rel) - raw_mrr)
            
            # v1
            def faiss_search_fn(qe, k=10):
                rk = faiss_search(qe, k)
                return rk, [1.0]*len(rk) # scores don't matter for mrr
            
            v1_res = setu_v1_fixed_order(
                query=q_text, raw_ranking=(r_rank, [1.0]*len(r_rank)), embed_fn=embed_fn,
                entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
                lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, lag_model=lag_model
            )
            v1_rank = v1_res["final_ranking"]
            deltas["SETU_v1"][group].append(get_mrr(v1_rank, rel) - raw_mrr)
            
    # Compute Statistics
    print(f"RAW already correct (MRR=1): {raw_correct_count} ({(raw_correct_count/len(queries))*100:.1f}%)")
    print(f"RAW incorrect (MRR<1): {raw_incorrect_count} ({(raw_incorrect_count/len(queries))*100:.1f}%)")
    
    model_results = {
        "group_sizes": {
            "RAW_already_correct": raw_correct_count,
            "percent_correct": round((raw_correct_count/len(queries))*100, 2),
            "RAW_incorrect": raw_incorrect_count,
            "percent_incorrect": round((raw_incorrect_count/len(queries))*100, 2)
        },
        "operators": {}
    }
    
    for op, grp_deltas in deltas.items():
        op_res = {}
        for grp in ["correct", "incorrect"]:
            d_list = grp_deltas[grp]
            if len(d_list) == 0:
                continue
            mean_d = float(np.mean(d_list))
            # Wilcoxon requires non-zero differences, otherwise it throws ValueError. Add safe fallback.
            non_zero_d = [x for x in d_list if abs(x) > 1e-6]
            if len(non_zero_d) > 0:
                try:
                    stat, p_val = wilcoxon(d_list)
                    p_val = float(p_val)
                except Exception:
                    p_val = 1.0
            else:
                p_val = 1.0
            
            op_res[f"mean_delta_{grp}"] = round(mean_d, 4)
            op_res[f"wilcoxon_p_{grp}"] = round(p_val, 4)
            op_res[f"significant_{grp}"] = p_val < 0.05
        model_results["operators"][op] = op_res
        
    final_results[model_key] = model_results
    plot_data[model_key] = deltas

# Save JSON
out_path = ROOT / "results" / "tables" / "overcorrection_final.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)

# Create Bar Chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (model_key, p_data) in enumerate(plot_data.items()):
    ax = axes[idx]
    ops = list(p_data.keys())
    x = np.arange(len(ops))
    width = 0.35
    
    correct_means = [np.mean(p_data[op]["correct"]) for op in ops]
    incorrect_means = [np.mean(p_data[op]["incorrect"]) for op in ops]
    
    rects1 = ax.bar(x - width/2, correct_means, width, label='RAW Correct (MRR=1)', color='salmon')
    rects2 = ax.bar(x + width/2, incorrect_means, width, label='RAW Incorrect (MRR<1)', color='skyblue')
    
    ax.set_ylabel('Mean Δ MRR')
    ax.set_title(f'Over-Correction Effect ({model_key})')
    ax.set_xticks(x)
    ax.set_xticklabels(ops)
    if idx == 0:
        ax.legend()
    ax.axhline(0, color='black', linewidth=0.8)

plt.tight_layout()
fig_path = ROOT / "results" / "figures" / "overcorrection.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=300)

print(f"\nSaved results to {out_path}")
print(f"Saved figure to {fig_path}")
