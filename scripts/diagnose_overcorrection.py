import json
import numpy as np
import faiss
import pickle
from pathlib import Path
import sys

ROOT = Path('.')
sys.path.append(str(ROOT))

from setu.operators.lqp import apply_lqp
from setu.operators.caep import apply_caep
from setu.operators.lag import apply_lag, entity_density, predict_strategy
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.fusion.carf import rrf_baseline
from sentence_transformers import SentenceTransformer

print("Loading data...")
queries = json.load(open(ROOT / 'data' / 'processed' / 'queries_v3_final.json', encoding='utf-8'))
per_query = json.load(open(ROOT / 'results' / 'logs' / 'per_query_metrics_v2.json', encoding='utf-8'))['bge_m3']
doc_emb = np.load(ROOT / 'results' / 'logs' / 'doc_emb_bge_m3_v2.npy')
q_emb_base = np.load(ROOT / 'results' / 'logs' / 'query_emb_bge_m3_v3.npy')

lqp_model = pickle.load(open(ROOT / 'results' / 'models' / 'lqp_model_bge_m3.pkl', 'rb'))
caep_gate = pickle.load(open(ROOT / 'results' / 'models' / 'caep_gate_bge_m3.pkl', 'rb'))
lag_model = pickle.load(open(ROOT / 'results' / 'models' / 'lag_model_v3.pkl', 'rb'))

print("Loading model for CAEP re-encoding...")
model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)

from setu.operators.caep import extract_entity_list, entity_frequencies
corpus_lines = [json.loads(line) for line in open(ROOT / 'data' / 'processed' / 'corpus_chunks_v2.jsonl', encoding='utf-8')]
docids = [d['chunk_id'] for d in corpus_lines]
doc_texts = [d['text'] for d in corpus_lines]
corpus_entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)

index = faiss.IndexFlatIP(doc_emb.shape[1])
index.add(doc_emb.astype(np.float32))

print("Running baseline FAISS to get baseline rankings (needed for LAG)...")
base_scores, base_indices = index.search(q_emb_base.astype(np.float32), 10)

def get_mrr(ranking, rel_docs):
    for r, did in enumerate(ranking):
        if did in rel_docs:
            return 1.0 / (r + 1)
    return 0.0

lqp_deltas = {'correct': [], 'wrong': []}
caep_deltas = {'correct': [], 'wrong': []}
lag_deltas = {'correct': [], 'wrong': []}

for i, q in enumerate(queries):
    qid = q['query_id']
    if qid not in per_query: continue
    
    raw_mrr = per_query[qid]['mrr']
    group = 'correct' if raw_mrr == 1.0 else 'wrong'
    rel_docs = set(q['relevant_doc_ids'])
    
    # LQP
    q_cmi = cmi(q['text'])
    l_emb = apply_lqp(q_emb_base[i], q_cmi, lqp_model)
    _, l_idx = index.search(np.array([l_emb]).astype(np.float32), 10)
    l_ranking = [docids[d] for d in l_idx[0]]
    lqp_mrr = get_mrr(l_ranking, rel_docs)
    lqp_deltas[group].append(lqp_mrr - raw_mrr)
    
    # LAG
    q_ent = lid_entropy(q['text'])
    q_dens = entity_density(q['text'], corpus_entities)
    lag_strat = predict_strategy(q_cmi, q_ent, q_dens, lag_model)
    
    def embed_fn(texts):
        return model.encode(texts, convert_to_numpy=True).astype("float32")
        
    lag_out = apply_lag(
        q['text'],
        lag_strat,
        entities=corpus_entities,
        embed_fn=embed_fn,
        caep_gate=caep_gate,
        entity_freq=entity_freq,
    )
    
    def faiss_search(query_emb, k=10):
        q_emb = np.asarray(query_emb, dtype="float32").reshape(1, -1)
        _, indices = index.search(q_emb, k)
        return [docids[i] for i in indices[0]]
        
    if isinstance(lag_out, list):
        q1_emb = embed_fn([lag_out[0]])[0]
        q2_emb = embed_fn([lag_out[1]])[0]
        r1_ids = faiss_search(q1_emb)
        r2_ids = faiss_search(q2_emb)
        lag_ranking = rrf_baseline([r1_ids, r2_ids])
    else:
        lag_emb = embed_fn([lag_out])[0]
        lag_ranking = faiss_search(lag_emb)
        
    lag_mrr = get_mrr(lag_ranking, rel_docs)
    lag_deltas[group].append(lag_mrr - raw_mrr)
    
    # CAEP
    b_ranking = [docids[d] for d in base_indices[i]]
    c_text = apply_caep(q['text'], corpus_entities, caep_gate)
    if c_text != q['text']:
        c_emb = model.encode([c_text], convert_to_numpy=True)[0]
        _, c_idx = index.search(np.array([c_emb]).astype(np.float32), 10)
        c_ranking = [docids[d] for d in c_idx[0]]
    else:
        c_ranking = b_ranking
    caep_mrr = get_mrr(c_ranking, rel_docs)
    caep_deltas[group].append(caep_mrr - raw_mrr)

results_dict = {
    "group_sizes": {
        "RAW_already_correct": len(lqp_deltas['correct']),
        "RAW_got_it_wrong": len(lqp_deltas['wrong'])
    },
    "operator_deltas": {
        "LQP": {
            "Effect_on_Correct": float(np.mean(lqp_deltas['correct'])),
            "Effect_on_Wrong": float(np.mean(lqp_deltas['wrong']))
        },
        "CAEP": {
            "Effect_on_Correct": float(np.mean(caep_deltas['correct'])),
            "Effect_on_Wrong": float(np.mean(caep_deltas['wrong']))
        },
        "LAG": {
            "Effect_on_Correct": float(np.mean(lag_deltas['correct'])),
            "Effect_on_Wrong": float(np.mean(lag_deltas['wrong']))
        }
    }
}

out_path = ROOT / 'results' / 'tables' / 'overcorrection_diagnosis.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=4)

print("\n=== Over-Correction Diagnosis (Mean MRR Change vs RAW) ===")
print(f"RAW already correct (MRR=1.0) group size: {len(lqp_deltas['correct'])}")
print(f"RAW got it wrong  (MRR<1.0) group size: {len(lqp_deltas['wrong'])}")

print("\nOperator | Effect on Correct | Effect on Wrong")
print("----------------------------------------------")
print(f"LQP      | {np.mean(lqp_deltas['correct']):+17.4f} | {np.mean(lqp_deltas['wrong']):+15.4f}")
print(f"CAEP     | {np.mean(caep_deltas['correct']):+17.4f} | {np.mean(caep_deltas['wrong']):+15.4f}")
print(f"LAG      | {np.mean(lag_deltas['correct']):+17.4f} | {np.mean(lag_deltas['wrong']):+15.4f}")
print(f"\nSaved results to {out_path}")
