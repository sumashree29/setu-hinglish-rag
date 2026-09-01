import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
import faiss
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, Ridge
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("c:/Users/sumas/Downloads/setu-hinglish-rag-skeleton/setu-hinglish-rag")
sys.path.append(str(ROOT))

import setu.config as config
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.operators.caep import extract_entity_list, entity_frequencies, build_entity_features, fit_caep_gate, apply_caep
from setu.operators.lqp import fit_lqp, apply_lqp, load_parallel_pairs_phinc
from setu.operators.lag import entity_density, predict_strategy, apply_lag, SUB_STRATEGIES, fit_lag_v2
from setu.fusion.carf import rrf_baseline

MODELS_DIR = ROOT / "results" / "models"
TABLE_DIR = ROOT / "results" / "tables"
DATA_DIR = ROOT / "data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data
corpus_v2_file = DATA_DIR / "processed" / "corpus_chunks_v2.jsonl"
queries_v3_file = DATA_DIR / "processed" / "queries_v3_final.json"

chunks_v2 = [json.loads(l) for l in open(corpus_v2_file, encoding='utf-8') if l.strip()]
queries_v3 = json.load(open(queries_v3_file, encoding='utf-8'))

doc_ids = [c["chunk_id"] for c in chunks_v2]
doc_texts = [c["text"] for c in chunks_v2]
query_ids = [q["query_id"] for q in queries_v3]
query_texts = [q["text"] for q in queries_v3]

print(f"Loaded {len(chunks_v2)} chunks and {len(queries_v3)} queries.")

# Compute CMI & diagnostic bands
def cmi_band(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return bands[-1][2]

query_cmi = {q["query_id"]: cmi(q["text"]) for q in queries_v3}
query_entropy = {q["query_id"]: lid_entropy(q["text"]) for q in queries_v3}
query_bands = {q["query_id"]: cmi_band(query_cmi[q["query_id"]], config.CMI_BANDS) for q in queries_v3}

# 2. Extract Entities & Frequency over the 380 Chunks
print("\n--- 1. Extracting Entities from 380 Corpus Chunks ---")
entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)
print(f"Extracted {len(entities)} unique domain entities from 380 chunks.")
print(f"Top 15 entities: {entities[:15]}")

# 3. Retrain LAG on Empirical Labels (314 Queries)
print("\n--- 2. Retraining LAG on 314 Queries (Empirical Labels) ---")

lag_labels_file = DATA_DIR / "processed" / "lag_labels_v3.json"
with open(lag_labels_file, "r", encoding="utf-8") as f:
    lag_labeled_data = json.load(f)

X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in lag_labeled_data])
y_lag = np.array([d["label"] for d in lag_labeled_data])

print(f"Fitting LAG model on {len(lag_labeled_data)} queries (Strategy distribution: {dict(zip(SUB_STRATEGIES, [sum(y_lag==0), sum(y_lag==1), sum(y_lag==2)]))})...")
lag_model = fit_lag_v2(X_lag, y_lag)
with open(MODELS_DIR / "lag_model_v3.pkl", "wb") as f:
    pickle.dump(lag_model, f)

# 4. Model-by-Model Operator Training & Standalone Ablation
EMBEDDING_MODELS = {
    "bge_m3": ("BAAI/bge-m3", False),
    "indic_sbert": ("l3cube-pune/indic-sentence-similarity-sbert", False),
    "me5_large": ("intfloat/multilingual-e5-large", True)
}

qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries_v3}
qrels_all = Qrels(qrels_dict)

ablation_results = {}
METRICS = ["recall@5", "recall@10", "mrr", "ndcg@10"]

# Prepare PHINC dataset once for LQP
print("\n--- 3. Loading PHINC Parallel Pairs for LQP ---")
try:
    from datasets import load_dataset
    phinc_ds = load_dataset("LingoIITGN/PHINC", split="train")
    phinc_ds = phinc_ds.select(range(min(500, len(phinc_ds))))
    hinglish_col = "Sentence" if "Sentence" in phinc_ds.column_names else "Hinglish Code-Mixed Sentence"
    english_col = "English_Translation" if "English_Translation" in phinc_ds.column_names else "Human Translated English Sentence"
    hinglish_sents = [row[hinglish_col] for row in phinc_ds]
    english_sents = [row[english_col] for row in phinc_ds]
    print(f"Loaded {len(hinglish_sents)} parallel pairs from PHINC.")
except Exception as e:
    print(f"Warning: Could not download PHINC dataset directly ({e}). Falling back to local pairs.")
    hinglish_sents = ["BSBDA account kaise open kare", "ATM se paise nikalne ki limit kya hai", "KYC documents kya chahiye"] * 100
    english_sents = ["How to open BSBDA account", "What is ATM cash withdrawal limit", "What KYC documents are required"] * 100

for model_key, (model_name, needs_prefix) in EMBEDDING_MODELS.items():
    print(f"\n=======================================================")
    print(f"Training Operators & Evaluating Ablation: {model_key} ({model_name})")
    print(f"=======================================================")
    
    st_model = SentenceTransformer(model_name)
    def embed_fn(texts):
        inp = [f"query: {t}" for t in texts] if needs_prefix else texts
        return st_model.encode(inp, convert_to_numpy=True).astype("float32")

    # A. Train LQP for this model (Bypassed! Load existing)
    print(f"Loading pre-trained LQP for {model_key}...")
    import pickle
    try:
        with open(MODELS_DIR / f"lqp_model_{model_key}.pkl", "rb") as f:
            lqp_model = pickle.load(f)
    except Exception as e:
        print(f"Warning: LQP model load failed: {e}")
        # fallback if somehow missing, though it shouldn't be
        X_phinc = embed_fn(hinglish_sents)
        Y_phinc = embed_fn(english_sents)
        lqp_model = fit_lqp(X_phinc, Y_phinc, alpha_reg=1.0)
        
    # B. Train CAEP Gate for this model (Bypassed! Load existing)
    print(f"Loading pre-trained CAEP Gate for {model_key}...")
    import pickle
    try:
        with open(MODELS_DIR / f"caep_gate_{model_key}.pkl", "rb") as f:
            caep_gate = pickle.load(f)
    except Exception as e:
        print(f"Warning: CAEP gate load failed, falling back to dummy gate: {e}")
        from setu.operators.caep import CAEPGate
        caep_gate = CAEPGate(0.85)

    # C. Load Doc Embeddings & Build FAISS Index
    doc_emb_path = DATA_DIR / "embeddings" / f"doc_emb_{model_key}_v2.npy"
    doc_emb = np.load(doc_emb_path)
    faiss.normalize_L2(doc_emb)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)
    
    def faiss_search(query_emb, k=10):
        q = np.asarray(query_emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, k)
        return [doc_ids[i] for i in indices[0]], [float(s) for s in scores[0]]

    # D. Evaluate Standalone Operators
    print(f"Running standalone operator ablation for {model_key} across {len(queries_v3)} queries...")
    raw_run, lqp_run, caep_run, lag_run = {}, {}, {}, {}
    lat_raw, lat_lqp, lat_caep, lat_lag = [], [], [], []

    for q in queries_v3:
        qid = q["query_id"]
        q_text = q["text"]
        q_c = query_cmi[qid]
        q_ent = query_entropy[qid]
        
        # 1. RAW
        t0 = time.perf_counter()
        raw_emb = embed_fn([q_text])[0]
        r_ids, r_sc = faiss_search(raw_emb)
        lat_raw.append(time.perf_counter() - t0)
        raw_run[qid] = {d: (len(r_ids) - rk) for rk, d in enumerate(r_ids)}
        
        # 2. LQP ALONE
        t0 = time.perf_counter()
        lqp_emb = apply_lqp(np.asarray(raw_emb), q_c, lqp_model, cmi_max=1.0)
        l_ids, _ = faiss_search(lqp_emb)
        lat_lqp.append(time.perf_counter() - t0)
        lqp_run[qid] = {d: (len(l_ids) - rk) for rk, d in enumerate(l_ids)}
        
        # 3. CAEP ALONE
        t0 = time.perf_counter()
        caep_text = apply_caep(q_text, entities, caep_gate, entity_freq, embed_fn=embed_fn)
        caep_emb = embed_fn([caep_text])[0]
        c_ids, _ = faiss_search(caep_emb)
        lat_caep.append(time.perf_counter() - t0)
        caep_run[qid] = {d: (len(c_ids) - rk) for rk, d in enumerate(c_ids)}
        
        # 4. LAG ALONE
        t0 = time.perf_counter()
        q_dens = entity_density(q_text, entities)
        lag_strat = predict_strategy(q_c, q_ent, q_dens, lag_model)
        lag_out = apply_lag(
            q_text,
            lag_strat,
            entities=entities,
            embed_fn=embed_fn,
            caep_gate=caep_gate,
            entity_freq=entity_freq,
        )
        if isinstance(lag_out, list):
            q1_emb = embed_fn([lag_out[0]])[0]
            q2_emb = embed_fn([lag_out[1]])[0]
            r1_ids, _ = faiss_search(np.asarray(q1_emb))
            r2_ids, _ = faiss_search(np.asarray(q2_emb))
            lag_ids = rrf_baseline([r1_ids, r2_ids])
        else:
            lag_emb = embed_fn([lag_out])[0]
            lag_ids, _ = faiss_search(np.asarray(lag_emb))
        lat_lag.append(time.perf_counter() - t0)
        lag_run[qid] = {d: (len(lag_ids) - rk) for rk, d in enumerate(lag_ids)}

    # Overall Evaluation
    run_objs = {
        "RAW": Run(raw_run),
        "LQP_alone": Run(lqp_run),
        "CAEP_alone": Run(caep_run),
        "LAG_alone": Run(lag_run)
    }
    
    eval_overall = {}
    for op_name, r_obj in run_objs.items():
        res = evaluate(qrels_all, r_obj, METRICS)
        eval_overall[op_name] = {k: float(v) for k, v in res.items()}
        
    eval_overall["RAW"]["latency_ms"] = float(np.mean(lat_raw) * 1000)
    eval_overall["LQP_alone"]["latency_ms"] = float(np.mean(lat_lqp) * 1000)
    eval_overall["CAEP_alone"]["latency_ms"] = float(np.mean(lat_caep) * 1000)
    eval_overall["LAG_alone"]["latency_ms"] = float(np.mean(lat_lag) * 1000)

    # Subgroup Evaluation by CMI Band
    subgroup_eval = {}
    for band_name in ["low", "medium", "high", "very_high"]:
        b_qids = set(qid for qid, b in query_bands.items() if b == band_name)
        if not b_qids: continue
        qrels_sub = Qrels({qid: qrels_dict[qid] for qid in b_qids})
        
        band_res = {}
        for op_name, r_obj in run_objs.items():
            sub_run_dict = {qid: r_obj[qid] for qid in b_qids}
            band_res[op_name] = {k: float(v) for k, v in evaluate(qrels_sub, Run(sub_run_dict), METRICS).items()}
        subgroup_eval[band_name] = band_res

    ablation_results[model_key] = {
        "overall_314": eval_overall,
        "by_cmi_band": subgroup_eval
    }

# 5. Save Ablation Table
out_table = TABLE_DIR / "scaled_operator_ablation_v3.json"
with open(out_table, "w", encoding="utf-8") as f:
    json.dump(ablation_results, f, indent=2)
print(f"\nSaved standalone operator ablation table to: {out_table}")

# 6. Display Formatted Tables
print("\n" + "="*100)
print("PHASE 5 STANDALONE OPERATOR ABLATION TABLE (380 CHUNKS, 314 QUERIES)")
print("="*100)
print(f"{'Model':<12} | {'Operator':<12} | {'Recall@5':<10} | {'Δ Recall@5':<12} | {'Recall@10':<10} | {'MRR':<10} | {'Δ MRR':<10} | {'nDCG@10':<10} | {'Latency'}")
print("-"*100)

for m_key in EMBEDDING_MODELS:
    m_res = ablation_results[m_key]["overall_314"]
    raw_r5 = m_res["RAW"]["recall@5"]
    raw_mrr = m_res["RAW"]["mrr"]
    
    for op in ["RAW", "LQP_alone", "CAEP_alone", "LAG_alone"]:
        r5 = m_res[op]["recall@5"]
        r10 = m_res[op]["recall@10"]
        mrr = m_res[op]["mrr"]
        ndcg = m_res[op]["ndcg@10"]
        lat = m_res[op]["latency_ms"]
        
        d_r5 = f"{r5 - raw_r5:+0.4f}" if op != "RAW" else "baseline"
        d_mrr = f"{mrr - raw_mrr:+0.4f}" if op != "RAW" else "baseline"
        
        print(f"{m_key:<12} | {op:<12} | {r5:<10.4f} | {d_r5:<12} | {r10:<10.4f} | {mrr:<10.4f} | {d_mrr:<10} | {ndcg:<10.4f} | {lat:5.1f}ms")
    print("-" * 100)

print("\n" + "="*100)
print("SUBGROUP ABLATION: LOW (n=14) vs VERY HIGH (n=35) CMI BANDS")
print("="*100)
for m_key in EMBEDDING_MODELS:
    print(f"\nModel: {m_key}")
    print(f"{'Band':<12} | {'Operator':<12} | {'Recall@5':<10} | {'Δ Recall@5':<12} | {'MRR':<10} | {'Δ MRR':<10} | {'nDCG@10':<10}")
    print("-"*80)
    for band in ["low", "very_high"]:
        b_res = ablation_results[m_key]["by_cmi_band"][band]
        b_raw_r5 = b_res["RAW"]["recall@5"]
        b_raw_mrr = b_res["RAW"]["mrr"]
        for op in ["RAW", "LQP_alone", "CAEP_alone", "LAG_alone"]:
            r5 = b_res[op]["recall@5"]
            mrr = b_res[op]["mrr"]
            ndcg = b_res[op]["ndcg@10"]
            d_r5 = f"{r5 - b_raw_r5:+0.4f}" if op != "RAW" else "baseline"
            d_mrr = f"{mrr - b_raw_mrr:+0.4f}" if op != "RAW" else "baseline"
            print(f"{band:<12} | {op:<12} | {r5:<10.4f} | {d_r5:<12} | {mrr:<10.4f} | {d_mrr:<10} | {ndcg:<10.4f}")
        print("-" * 80)
