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
from setu.operators.lag import apply_lag
from setu.diagnosis.cmi import cmi
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

from setu.operators.caep import extract_entity_list
corpus_lines = [json.loads(line) for line in open(ROOT / 'data' / 'processed' / 'corpus_chunks_v2.jsonl', encoding='utf-8')]
docids = [d['chunk_id'] for d in corpus_lines]
corpus_entities = extract_entity_list([d['text'] for d in corpus_lines])

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
    b_ranking = [docids[d] for d in base_indices[i]]
    scores_list = base_scores[i].tolist()
    lag_ranking = apply_lag(q['text'], b_ranking, scores_list, lag_model)
    lag_mrr = get_mrr(lag_ranking, rel_docs)
    lag_deltas[group].append(lag_mrr - raw_mrr)
    
    # CAEP
    c_text = apply_caep(q['text'], corpus_entities, caep_gate)
    if c_text != q['text']:
        c_emb = model.encode([c_text], convert_to_numpy=True)[0]
        _, c_idx = index.search(np.array([c_emb]).astype(np.float32), 10)
        c_ranking = [docids[d] for d in c_idx[0]]
    else:
        c_ranking = b_ranking
    caep_mrr = get_mrr(c_ranking, rel_docs)
    caep_deltas[group].append(caep_mrr - raw_mrr)

print("\n=== Over-Correction Diagnosis (Mean MRR Change vs RAW) ===")
print(f"RAW already correct (MRR=1.0) group size: {len(lqp_deltas['correct'])}")
print(f"RAW got it wrong  (MRR<1.0) group size: {len(lqp_deltas['wrong'])}")

print("\nOperator | Effect on Correct | Effect on Wrong")
print("----------------------------------------------")
print(f"LQP      | {np.mean(lqp_deltas['correct']):+17.4f} | {np.mean(lqp_deltas['wrong']):+15.4f}")
print(f"CAEP     | {np.mean(caep_deltas['correct']):+17.4f} | {np.mean(caep_deltas['wrong']):+15.4f}")
print(f"LAG      | {np.mean(lag_deltas['correct']):+17.4f} | {np.mean(lag_deltas['wrong']):+15.4f}")
