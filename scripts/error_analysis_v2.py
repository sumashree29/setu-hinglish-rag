"""
Phase 6: CMI-stratified error analysis of SETU v2 failures.
"""
import json
import sys
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Set seed for reproducibility
np.random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from setu.config import CMI_BANDS
from setu.diagnosis.cmi import cmi
from setu.operators.caep import extract_entity_list, entity_frequencies
from setu.controller.setu_bandit import setu_v2_run, LinUCBController, setu_v1_fixed_order
from setu.evaluation.metrics import confidence_proxy

def cmi_band(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return bands[-1][2]

def main():
    print("Loading data and models...")
    queries = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", encoding="utf-8"))
    chunks = [json.loads(line) for line in open(ROOT / "data" / "processed" / "corpus_chunks_v2.jsonl", encoding="utf-8") if line.strip()]
    
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    
    model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
    doc_emb = model.encode(doc_texts, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(doc_emb)
    idx = faiss.IndexFlatIP(doc_emb.shape[1])
    idx.add(doc_emb)

    def faiss_search(q_emb, k=10):
        q = np.asarray(q_emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        s, ind = idx.search(q, k)
        return [doc_ids[i] for i in ind[0]], [float(score) for score in s[0]]

    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)
    
    caep_gate = pickle.load(open(ROOT / "results" / "models" / "caep_gate_bge_m3.pkl", "rb"))
    lqp_model = pickle.load(open(ROOT / "results" / "models" / "lqp_model_bge_m3.pkl", "rb"))
    lag_model = pickle.load(open(ROOT / "results" / "models" / "lag_model_v3.pkl", "rb"))

    # Load clean trajectories to build OOF controllers just like in run_statistical_tests_h1_h10_scaled.py
    trajectories = [json.loads(line) for line in open(ROOT / "data" / "logs" / "trajectories.jsonl", encoding="utf-8") if line.strip()]
    n_splits = 5
    qids = [q["query_id"] for q in queries]
    q_by_id = {q["query_id"]: q for q in queries}
    folds = np.array_split(qids, n_splits)
    
    fold_ctrls = {}
    for f_idx, t_qids in enumerate(folds):
        t_texts = set(q_by_id[qid]["text"] for qid in t_qids)
        tr_traj = [r for r in trajectories if r.get("query") not in t_texts]
        f_ctrl = LinUCBController(context_dim=7, alpha=0.0)
        cur_q = None
        tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
        for row in tr_traj:
            q_t = row.get("query")
            step_v = float(row.get("state", {}).get("step", 0))
            if q_t != cur_q or step_v == 0:
                cur_q = q_t
                tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
            c_v = float(row["state"]["cmi"])
            e_v = float(row["state"]["lid_entropy"])
            cf_v = float(row["state"]["confidence"])
            ctx = np.array([c_v, e_v, cf_v, step_v, tried["LAG"], tried["CAEP"], tried["LQP"]], dtype=float)
            f_ctrl.update(ctx, row["action"], float(row.get("reward", 0.0)))
            if row["action"] in tried: tried[row["action"]] = 1.0
        for qid in t_qids:
            fold_ctrls[qid] = f_ctrl

    print("Running evaluation...")
    failures = []
    
    for q in queries:
        qid = q["query_id"]
        q_txt = q["text"]
        rel_docs = set(q["relevant_doc_ids"])
        q_cmi = cmi(q_txt)
        q_band = cmi_band(q_cmi, CMI_BANDS)
        
        q_emb = model.encode([q_txt], convert_to_numpy=True)[0]
        raw_ranking = faiss_search(q_emb)
        raw_mrr = 1.0 if raw_ranking[0][0] in rel_docs else 0.0
        
        v1_res = setu_v1_fixed_order(
            query=q_txt, raw_ranking=raw_ranking, embed_fn=lambda t: model.encode(t, convert_to_numpy=True),
            entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
            lqp_model=lqp_model, faiss_search_fn=faiss_search, lag_model=lag_model,
        )
        v1_mrr = 1.0 if v1_res["final_ranking"][0] in rel_docs else 0.0
        
        ops, conf_tr, v2_rank = setu_v2_run(
            query=q_txt, controller=fold_ctrls[qid], raw_ranking=raw_ranking,
            embed_fn=lambda t: model.encode(t, convert_to_numpy=True),
            entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
            lqp_model=lqp_model, faiss_search_fn=faiss_search, confidence_fn=confidence_proxy,
            lag_model=lag_model, train=False,
        )
        
        v2_mrr = 1.0 if v2_rank[0] in rel_docs else 0.0
        
        if v2_mrr < 1.0:
            ops_no_stop = [o for o in ops if o != "STOP"]
            
            # Categorize
            category = "unexplained"
            if raw_mrr < 1.0 and not ops_no_stop:
                category = "base_retrieval_miss"
            elif "CAEP" in ops_no_stop:
                category = "caep_overcorrect" if raw_mrr == 1.0 else "caep_undercorrect"
            elif "LAG" in ops_no_stop:
                category = "lag_wrong_substrategy"
            elif "LQP" in ops_no_stop:
                category = "lqp_noop"
            elif raw_mrr < 1.0:
                category = "base_retrieval_miss"
                
            failures.append({
                "query_id": qid,
                "text": q_txt,
                "cmi": q_cmi,
                "band": q_band,
                "ops": ops_no_stop,
                "raw_failed": raw_mrr < 1.0,
                "v1_failed": v1_mrr < 1.0,
                "category": category
            })

    # Aggregation
    band_counts = {b[2]: 0 for b in CMI_BANDS}
    category_by_band = {b[2]: {} for b in CMI_BANDS}
    
    for f in failures:
        b = f["band"]
        c = f["category"]
        band_counts[b] += 1
        category_by_band[b][c] = category_by_band[b].get(c, 0) + 1
        
    output = {
        "total_queries": len(queries),
        "total_v2_failures_at_rank1": len(failures),
        "band_counts": band_counts,
        "category_breakdown_by_band": category_by_band,
        "examples": {}
    }
    
    # 3-5 examples per category
    examples_by_cat = {}
    for f in failures:
        c = f["category"]
        if c not in examples_by_cat:
            examples_by_cat[c] = []
        if len(examples_by_cat[c]) < 5:
            examples_by_cat[c].append({
                "query": f["text"],
                "cmi": round(f["cmi"], 3),
                "ops_applied": f["ops"],
                "raw_failed": f["raw_failed"]
            })
    
    output["examples"] = examples_by_cat

    out_path1 = ROOT / "results" / "tables" / "error_analysis_v2.json"
    out_path2 = ROOT / "outputs" / "tables" / "error_analysis_v2.json"
    
    for p in [out_path1, out_path2]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            
    print(f"\nAnalysis complete! {len(failures)} failures found.")
    print(f"Results saved to {out_path1} and {out_path2}")
    
    print("\n--- Failure Breakdown by Band ---")
    for b in band_counts:
        print(f"{b}: {band_counts[b]} total")
        for c, count in category_by_band[b].items():
            print(f"  - {c}: {count} ({count/band_counts[b]*100:.1f}%)" if band_counts[b] else f"  - {c}: 0")

if __name__ == "__main__":
    main()
