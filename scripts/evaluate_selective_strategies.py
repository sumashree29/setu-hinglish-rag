"""
Evaluate all four selective-correction reference strategies on BGE-M3 (314 queries, 380 chunks):
  1. pi_never:    RAW baseline (no modification)
  2. pi_always:   SETU_v1 (fixed LAG -> CAEP -> LQP -> CARF pipeline)
  3. pi_learned:  SETU_v2 (grouped 5-fold OOF LinUCB controller)
  4. pi_threshold: Score-margin confidence gating (trigger corrective fallback only if margin < tau)
  5. pi_star:     Empirical Oracle (max MRR per query across candidate executions)
Persists canonical outputs to results/tables/selective_correction_strategies.json.
"""
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from setu.config import set_seed
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.embeddings.loader import load_embedding_model, embed
from setu.evaluation.metrics import confidence_proxy
from setu.fusion.carf import rrf_baseline
from setu.operators.caep import extract_entity_list, entity_frequencies, apply_caep
from setu.operators.lag import fit_lag_v2, predict_strategy, apply_lag, entity_density
from setu.operators.lqp import apply_lqp
from setu.controller.setu_bandit import setu_v1_fixed_order, setu_v2_run, LinUCBController
from scripts.compare_setu_v1_v2_scaled import build_grouped_folds, split_trajectories_by_fold


def get_mrr(ranking, rel_docs):
    for r, did in enumerate(ranking):
        if did in rel_docs:
            return 1.0 / (r + 1)
    return 0.0


def main():
    set_seed()
    chunks = [json.loads(l) for l in open(ROOT / "data/processed/corpus_chunks_v2.jsonl", encoding="utf-8") if l.strip()]
    queries = json.load(open(ROOT / "data/processed/queries_v3_final.json", encoding="utf-8"))
    lag_labels_data = json.load(open(ROOT / "data/processed/lag_labels_v3.json", encoding="utf-8"))

    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    qrels = {q["query_id"]: set(q["relevant_doc_ids"]) for q in queries}
    q_by_id = {q["query_id"]: q for q in queries}
    qids = [q["query_id"] for q in queries]

    st_model = load_embedding_model("bge_m3")

    embed_cache = {}

    def embed_fn(texts):
        if len(texts) <= 4 and all(t in embed_cache for t in texts):
            return np.stack([embed_cache[t] for t in texts])
        vecs = embed(texts, st_model).astype("float32")
        for t, v in zip(texts, vecs):
            if len(embed_cache) < 5000:
                embed_cache[t] = v
        return vecs

    doc_emb_path = ROOT / "results/logs/doc_emb_bge_m3_v2.npy"
    doc_emb = np.load(doc_emb_path).astype("float32") if doc_emb_path.exists() else embed_fn(doc_texts)
    faiss.normalize_L2(doc_emb)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    index.add(doc_emb)

    def faiss_search_fn(query_embedding, k=10):
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, k)
        return [doc_ids[i] for i in indices[0]], [float(s) for s in scores[0]]

    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)
    caep_gate = pickle.load(open(ROOT / "results/models/caep_gate_bge_m3.pkl", "rb"))
    lqp_model = pickle.load(open(ROOT / "results/models/lqp_model_bge_m3.pkl", "rb"))

    trajectories = [json.loads(l) for l in open(ROOT / "data/logs/trajectories_v3.jsonl", encoding="utf-8") if l.strip()]
    folds = build_grouped_folds(queries, n_splits=5)

    per_query = {}
    for test_qids in folds:
        train_ids = set(qids) - set(test_qids)
        test_ids = set(test_qids)
        train_traj, _ = split_trajectories_by_fold(trajectories, train_ids, test_ids)

        fold_lag_data = [d for d in lag_labels_data if d["query_id"] in train_ids]
        X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in fold_lag_data])
        y_lag = np.array([d["label"] for d in fold_lag_data])
        fold_lag_model = fit_lag_v2(X_lag, y_lag)

        fold_ctrl = LinUCBController(context_dim=7, alpha=0.0)
        cur_qid = None
        tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
        for row in train_traj:
            rqid = row.get("query_id")
            step_val = float(row.get("state", {}).get("step", 0))
            if rqid != cur_qid or step_val == 0:
                cur_qid = rqid
                tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
            ctx = np.array([
                float(row["state"]["cmi"]),
                float(row["state"]["lid_entropy"]),
                float(row["state"]["confidence"]),
                step_val,
                tried["LAG"], tried["CAEP"], tried["LQP"],
            ], dtype=float)
            act = row["action"]
            fold_ctrl.update(ctx, act, float(row.get("reward", 0.0)))
            if act in tried:
                tried[act] = 1.0

        for qid in test_qids:
            q_text = q_by_id[qid]["text"]
            rel = qrels[qid]
            cmi_val = cmi(q_text)
            ent_val = lid_entropy(q_text)
            dens_val = entity_density(q_text, entities)

            raw_emb = embed_fn([q_text])[0]
            raw_rank, raw_scores = faiss_search_fn(raw_emb)
            margin = confidence_proxy(raw_scores, method="margin")

            # LQP
            lqp_emb = apply_lqp(raw_emb, cmi_val, lqp_model, cmi_max=1.0)
            lqp_rank, _ = faiss_search_fn(lqp_emb)

            # CAEP
            caep_txt = apply_caep(q_text, entities, caep_gate, entity_freq, embed_fn=embed_fn)
            caep_rank = raw_rank if caep_txt == q_text else faiss_search_fn(embed_fn([caep_txt])[0])[0]

            # LAG
            strat = predict_strategy(cmi_val, ent_val, dens_val, fold_lag_model)
            lag_out = apply_lag(q_text, strat, entities=entities, embed_fn=embed_fn, caep_gate=caep_gate, entity_freq=entity_freq)
            if isinstance(lag_out, list):
                r1, _ = faiss_search_fn(embed_fn([lag_out[0]])[0])
                r2, _ = faiss_search_fn(embed_fn([lag_out[1]])[0])
                lag_rank = rrf_baseline([r1, r2])
            else:
                lag_rank, _ = faiss_search_fn(embed_fn([lag_out])[0])

            # SETU_v1 & SETU_v2
            v1_res = setu_v1_fixed_order(
                query=q_text, raw_ranking=(raw_rank, raw_scores), embed_fn=embed_fn,
                entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
                lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, lag_model=fold_lag_model,
            )
            ops_v2, _, v2_rank, _ = setu_v2_run(
                query=q_text, query_id=qid, controller=fold_ctrl, raw_ranking=(raw_rank, raw_scores),
                embed_fn=embed_fn, entities=entities, entity_freq=entity_freq,
                caep_gate=caep_gate, lqp_model=lqp_model, faiss_search_fn=faiss_search_fn,
                confidence_fn=confidence_proxy, lag_model=fold_lag_model, train=False,
            )

            mrrs = {
                "RAW": get_mrr(raw_rank, rel),
                "LQP": get_mrr(lqp_rank, rel),
                "CAEP": get_mrr(caep_rank, rel),
                "LAG": get_mrr(lag_rank, rel),
                "SETU_v1": get_mrr(v1_res["final_ranking"], rel),
                "SETU_v2": get_mrr(v2_rank, rel),
            }
            best_op = max(mrrs, key=mrrs.get)
            per_query[qid] = {
                "margin": float(margin),
                "mrrs": mrrs,
                "oracle_best_op": best_op,
                "oracle_mrr": float(mrrs[best_op]),
                "v2_steps": len([o for o in ops_v2 if o != "STOP"]),
            }

    raw_arr = np.array([per_query[qid]["mrrs"]["RAW"] for qid in qids])
    strategies = {
        "pi_never_RAW": raw_arr,
        "pi_always_SETU_v1": np.array([per_query[qid]["mrrs"]["SETU_v1"] for qid in qids]),
        "pi_learned_SETU_v2": np.array([per_query[qid]["mrrs"]["SETU_v2"] for qid in qids]),
    }

    thresholds = [0.02, 0.05, 0.10, 0.15]
    threshold_meta = {}
    for tau in thresholds:
        for fallback in ["LAG", "SETU_v2"]:
            key = f"pi_threshold_tau_{tau:.2f}_{fallback}"
            vals = []
            triggered = 0
            for qid in qids:
                if per_query[qid]["margin"] < tau:
                    vals.append(per_query[qid]["mrrs"][fallback])
                    triggered += 1
                else:
                    vals.append(per_query[qid]["mrrs"]["RAW"])
            strategies[key] = np.array(vals)
            threshold_meta[key] = {
                "tau": tau,
                "fallback": fallback,
                "triggered_count": triggered,
                "triggered_pct": round(100.0 * triggered / len(qids), 2),
            }

    oracle_arr = np.array([per_query[qid]["oracle_mrr"] for qid in qids])
    strategies["pi_star_Oracle"] = oracle_arr

    # Paired Wilcoxon tests vs RAW with Holm-Bonferroni correction
    test_keys = [k for k in strategies if k != "pi_never_RAW"]
    raw_pvals = []
    for k in test_keys:
        diff = strategies[k] - raw_arr
        if np.any(np.abs(diff) > 1e-8):
            _, p = wilcoxon(diff)
            raw_pvals.append(float(p))
        else:
            raw_pvals.append(1.0)
    _, holm_pvals, _, _ = multipletests(raw_pvals, alpha=0.05, method="holm")

    summary = {}
    raw_mean = float(np.mean(raw_arr))
    for k, arr in strategies.items():
        entry = {
            "mrr": round(float(np.mean(arr)), 4),
            "delta_vs_raw": round(float(np.mean(arr)) - raw_mean, 4),
        }
        if k in threshold_meta:
            entry.update(threshold_meta[k])
        if k in test_keys:
            idx = test_keys.index(k)
            entry["wilcoxon_p_vs_raw"] = round(raw_pvals[idx], 6)
            entry["holm_p_vs_raw"] = round(float(holm_pvals[idx]), 6)
        summary[k] = entry

    oracle_counts = {}
    for qid in qids:
        op = per_query[qid]["oracle_best_op"]
        oracle_counts[op] = oracle_counts.get(op, 0) + 1

    out = {
        "n_queries": len(qids),
        "oracle_headroom_gap": round(float(np.mean(oracle_arr)) - raw_mean, 4),
        "margin_below_0.05_count": sum(1 for qid in qids if per_query[qid]["margin"] < 0.05),
        "margin_below_0.20_count": sum(1 for qid in qids if per_query[qid]["margin"] < 0.20),
        "oracle_selected_operator_counts": oracle_counts,
        "strategies": summary,
    }

    out_path = ROOT / "results/tables/selective_correction_strategies.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved selective correction evaluation to {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
