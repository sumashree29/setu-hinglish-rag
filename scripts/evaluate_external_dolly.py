"""
External Validation Arm (Phase 3):
Evaluates RAW dense baselines vs standalone corrective operators (+LQP, +CAEP, +LAG)
on 500 Hinglish QA pairs from aaditya/databricks-dolly-15k-Hinglish-Codemix
(zero overlap with RBI/PM-KISAN corpus and zero overlap with PHINC used to train LQP).
"""
import gc
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from ranx import Qrels, Run, evaluate
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from scripts.import_research_dataset import import_dolly_hinglish
from scripts.train_all_operator_models import build_caep_training_data
from setu.config import set_seed
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.embeddings.loader import load_embedding_model, embed
from setu.fusion.carf import rrf_baseline
from setu.operators.caep import extract_entity_list, entity_frequencies, fit_caep_gate, apply_caep
from setu.operators.lag import SUB_STRATEGIES, entity_density, light_normalize, dual_variant, full_rewrite, fit_lag_v2, predict_strategy, apply_lag
from setu.operators.lqp import apply_lqp


def get_mrr(ranking, rel_docs):
    for r, did in enumerate(ranking):
        if did in rel_docs:
            return 1.0 / (r + 1)
    return 0.0


def main():
    set_seed()
    out_dir = ROOT / "data" / "research_dolly_hinglish"
    corpus_file = out_dir / "corpus_chunks.jsonl"
    query_file = out_dir / "queries.json"
    if not (corpus_file.exists() and query_file.exists()) or corpus_file.stat().st_size < 10:
        corpus_file, query_file = import_dolly_hinglish(max_samples=500, out_dir=out_dir)

    chunks = [json.loads(l) for l in open(corpus_file, encoding="utf-8") if l.strip()]
    queries = json.load(open(query_file, encoding="utf-8"))
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)
    top_entities = sorted(entities, key=lambda e: entity_freq.get(e, 0), reverse=True)[:50]

    qrels_dict = {q["query_id"]: {d: 1 for d in q["gold_chunk_ids"]} for q in queries}
    qrels_obj = Qrels(qrels_dict)
    qids = [q["query_id"] for q in queries]
    q_by_id = {q["query_id"]: q for q in queries}

    backbones = {
        "bge_m3": False,
        "me5_large": True,
        "indic_sbert": False,
    }

    results = {
        "dataset": "aaditya/databricks-dolly-15k-Hinglish-Codemix",
        "n_passages": len(chunks),
        "n_queries": len(queries),
        "models": {},
    }
    all_pvals = []
    pval_refs = []

    for model_key, needs_prefix in backbones.items():
        print(f"\n=== External Validation: {model_key} on Dolly-Hinglish ({len(queries)} queries) ===")
        st_model = load_embedding_model(model_key)

        embed_cache = {}

        def embed_q(texts):
            missing = [t for t in dict.fromkeys(texts) if t not in embed_cache]
            if missing:
                inp = [f"query: {t}" for t in missing] if needs_prefix else missing
                vecs = embed(inp, st_model).astype("float32")
                for t, v in zip(missing, vecs):
                    embed_cache[t] = v
            return np.stack([embed_cache[t] for t in texts])

        def embed_d(texts):
            inp = [f"passage: {t}" for t in texts] if needs_prefix else texts
            return embed(inp, st_model).astype("float32")

        doc_emb = embed_d(doc_texts)
        faiss.normalize_L2(doc_emb)
        index = faiss.IndexFlatIP(doc_emb.shape[1])
        index.add(doc_emb)

        def search_fn(q_vec, k=10):
            q = np.asarray(q_vec, dtype="float32").reshape(1, -1)
            faiss.normalize_L2(q)
            _, idx = index.search(q, k)
            return [doc_ids[i] for i in idx[0]]

        # Load LQP (trained on PHINC, completely disjoint from Dolly-Hinglish)
        lqp_model = pickle.load(open(ROOT / f"results/models/lqp_model_{model_key}.pkl", "rb"))
        # Fit CAEP gate on Dolly-Hinglish corpus entities (no query leakage)
        caep_feats, caep_labs = build_caep_training_data(top_entities, entity_freq, embed_q)
        caep_gate = fit_caep_gate(caep_feats, caep_labs)

        # Compute RAW, LQP, CAEP + empirical best LAG labels for 5-fold OOF LAG evaluation
        q_texts = [q["query"] for q in queries]
        q_embs = embed_q(q_texts)

        raw_run, lqp_run, caep_run = {}, {}, {}
        lag_records = []
        strat_rankings = {}
        raw_mrrs = {}
        lqp_mrrs, caep_mrrs = {}, {}

        for i, q in enumerate(queries):
            if (i + 1) % 100 == 0:
                print(f"  [{model_key}] Encoded/evaluated {i + 1}/{len(queries)} queries...", flush=True)
            qid = q["query_id"]
            qt = q["query"]
            rel = set(q["gold_chunk_ids"])
            q_cmi = cmi(qt)
            q_ent = lid_entropy(qt)
            q_dens = entity_density(qt, entities)

            r_rank = search_fn(q_embs[i])
            raw_run[qid] = {d: (10 - rk) for rk, d in enumerate(r_rank)}
            raw_mrrs[qid] = get_mrr(r_rank, rel)

            l_rank = search_fn(apply_lqp(q_embs[i], q_cmi, lqp_model, cmi_max=1.0))
            lqp_run[qid] = {d: (10 - rk) for rk, d in enumerate(l_rank)}
            lqp_mrrs[qid] = get_mrr(l_rank, rel)

            c_txt = apply_caep(qt, entities, caep_gate, entity_freq, embed_fn=embed_q)
            c_rank = r_rank if c_txt == qt else search_fn(embed_q([c_txt])[0])
            caep_run[qid] = {d: (10 - rk) for rk, d in enumerate(c_rank)}
            caep_mrrs[qid] = get_mrr(c_rank, rel)

            # Empirical LAG label (OOF trained below)
            strat_scores = {}
            s_ln = search_fn(embed_q([light_normalize(qt)])[0])
            strat_scores["light_normalize"] = get_mrr(s_ln, rel)
            dv = dual_variant(qt, entities)
            s_dv = rrf_baseline([search_fn(embed_q([dv[0]])[0]), search_fn(embed_q([dv[1]])[0])])
            strat_scores["dual_variant"] = get_mrr(s_dv, rel)
            fr = full_rewrite(qt, entities=entities, caep_gate=caep_gate, entity_freq=entity_freq, embed_fn=embed_q)
            s_fr = search_fn(embed_q([fr])[0])
            strat_scores["full_rewrite"] = get_mrr(s_fr, rel)
            strat_rankings[i] = {"light_normalize": s_ln, "dual_variant": s_dv, "full_rewrite": s_fr}
            best_s = max(SUB_STRATEGIES, key=lambda s: strat_scores[s])
            lag_records.append({
                "query_id": qid,
                "cmi": q_cmi,
                "lid_entropy": q_ent,
                "entity_density": q_dens,
                "label": SUB_STRATEGIES.index(best_s),
            })

        # 5-fold OOF LAG evaluation
        lag_run, lag_mrrs = {}, {}
        kf = KFold(n_splits=5, shuffle=False)
        for train_idx, test_idx in kf.split(queries):
            X_tr = np.array([[lag_records[idx]["cmi"], lag_records[idx]["lid_entropy"], lag_records[idx]["entity_density"]] for idx in train_idx])
            y_tr = np.array([lag_records[idx]["label"] for idx in train_idx])
            fold_lag = fit_lag_v2(X_tr, y_tr) if len(set(y_tr.tolist())) >= 2 else None
            for idx in test_idx:
                q = queries[idx]
                qid, qt = q["query_id"], q["query"]
                rec = lag_records[idx]
                strat = predict_strategy(rec["cmi"], rec["lid_entropy"], rec["entity_density"], fold_lag) if fold_lag is not None else SUB_STRATEGIES[int(y_tr[0])]
                rk = strat_rankings[idx][strat]
                lag_run[qid] = {d: (10 - r) for r, d in enumerate(rk)}
                lag_mrrs[qid] = get_mrr(rk, set(q["gold_chunk_ids"]))

        METRICS = ["mrr", "ndcg@10", "hit_rate@1", "recall@10"]
        overall = {
            op_name: {k: round(float(v), 4) for k, v in evaluate(qrels_obj, Run(r_dict), METRICS).items()}
            for op_name, r_dict in [("RAW", raw_run), ("LQP_alone", lqp_run), ("CAEP_alone", caep_run), ("LAG_alone", lag_run)]
        }

        correct_qids = [qid for qid in qids if raw_mrrs[qid] == 1.0]
        incorrect_qids = [qid for qid in qids if raw_mrrs[qid] < 1.0]
        overcorr = {}
        for op_name, op_mrrs in [("LQP", lqp_mrrs), ("CAEP", caep_mrrs), ("LAG", lag_mrrs)]:
            op_entry = {}
            for grp_name, grp_qids in [("correct", correct_qids), ("incorrect", incorrect_qids)]:
                deltas = [op_mrrs[qid] - raw_mrrs[qid] for qid in grp_qids]
                op_entry[f"mean_delta_{grp_name}"] = round(float(np.mean(deltas)), 4) if deltas else 0.0
                if any(abs(x) > 1e-6 for x in deltas):
                    _, p_val = wilcoxon(deltas)
                    p_val = float(p_val)
                else:
                    p_val = 1.0
                all_pvals.append(p_val)
                pval_refs.append((model_key, op_name, grp_name))
                op_entry[f"wilcoxon_p_{grp_name}_raw"] = round(p_val, 4)
            overcorr[op_name] = op_entry

        results["models"][model_key] = {
            "overall": overall,
            "group_sizes": {
                "RAW_already_correct": len(correct_qids),
                "percent_correct": round(100.0 * len(correct_qids) / len(qids), 2),
                "RAW_incorrect": len(incorrect_qids),
                "percent_incorrect": round(100.0 * len(incorrect_qids) / len(qids), 2),
            },
            "overcorrection": overcorr,
        }

        del st_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _, holm_p, _, _ = multipletests(all_pvals, alpha=0.05, method="holm")
    for p_corr, (mk, op_n, grp_n) in zip(holm_p, pval_refs):
        results["models"][mk]["overcorrection"][op_n][f"wilcoxon_p_{grp_n}_holm"] = round(float(p_corr), 4)
        results["models"][mk]["overcorrection"][op_n][f"significant_{grp_n}_holm"] = bool(p_corr < 0.05)

    out_path = ROOT / "results/tables/external_dolly_validation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved external validation results to {out_path}")


if __name__ == "__main__":
    main()
