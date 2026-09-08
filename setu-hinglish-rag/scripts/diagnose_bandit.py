import json
import sys
from pathlib import Path
import numpy as np
from collections import Counter
sys.path.append(str(Path(__file__).resolve().parents[1]))
from setu.controller.setu_bandit import LinUCBController

# 1. Reward distribution
rewards = []
actions = []
with open("data/logs/trajectories.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            row = json.loads(line)
            rewards.append(float(row.get("reward", 0.0)))
            actions.append(row.get("action"))

rewards = np.array(rewards)
print("=== STEP 1: REWARD DISTRIBUTION ===")
print(f"Total rows: {len(rewards)}")
print(f"Mean reward: {np.mean(rewards):.6f}")
print(f"Std reward:  {np.std(rewards):.6f}")
print(f"Min reward:  {np.min(rewards):.6f}")
print(f"Max reward:  {np.max(rewards):.6f}")
print(f"Zero count:  {np.sum(rewards == 0)} ({np.mean(rewards == 0)*100:.1f}%)")
print(f"Pos count:   {np.sum(rewards > 0)}")
print(f"Neg count:   {np.sum(rewards < 0)}")
print("Action counts:", Counter(actions))

for act in sorted(set(actions)):
    mask = (np.array(actions) == act)
    act_rew = rewards[mask]
    print(f"  {act:5s}: mean={np.mean(act_rew):.6f}, std={np.std(act_rew):.6f}, min={np.min(act_rew):.6f}, max={np.max(act_rew):.6f}, non_zero={np.sum(act_rew != 0)}/{len(act_rew)}")

# 2. Compare Fold 0 vs Fold 1 Theta
queries = json.load(open("data/processed/queries_remapped.json", encoding="utf-8"))
trajectories = [json.loads(line) for line in open("data/logs/trajectories.jsonl", encoding="utf-8") if line.strip()]

n_splits = 5
qids = [q["query_id"] for q in queries]
q_by_id = {q["query_id"]: q for q in queries}
folds = np.array_split(qids, n_splits)

def train_fold(held_out_qids):
    held_out_texts = set(q_by_id[qid]["text"] for qid in held_out_qids)
    train_traj = [r for r in trajectories if r.get("query") not in held_out_texts]
    ctrl = LinUCBController(context_dim=7, alpha=0.0)
    current_query = None
    tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
    for row in train_traj:
        q = row.get("query")
        step_val = float(row.get("state", {}).get("step", 0))
        if q != current_query or step_val == 0:
            current_query = q
            tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
        cmi_val = float(row["state"]["cmi"])
        entropy_val = float(row["state"]["lid_entropy"])
        conf_val = float(row["state"]["confidence"])
        context = np.array([cmi_val, entropy_val, conf_val, step_val, tried["LAG"], tried["CAEP"], tried["LQP"]], dtype=float)
        action = row["action"]
        reward = float(row.get("reward", 0.0))
        ctrl.update(context, action, reward)
        if action in tried:
            tried[action] = 1.0
    return ctrl

ctrl0 = train_fold(folds[0])
ctrl1 = train_fold(folds[1])

print("\n=== STEP 1: THETA COMPARISON (FOLD 0 vs FOLD 1) ===")
for a in ["LAG", "CAEP", "LQP", "STOP"]:
    t0 = np.linalg.inv(ctrl0.A[a]) @ ctrl0.b[a]
    t1 = np.linalg.inv(ctrl1.A[a]) @ ctrl1.b[a]
    diff = np.linalg.norm(t0 - t1)
    print(f"Action {a:5s}: norm(theta0 - theta1) = {diff:.6f}")
    print(f"  Fold 0 theta: {np.round(t0, 4)}")
    print(f"  Fold 1 theta: {np.round(t1, 4)}")

# 3. Check select_action() logic
print("\n=== STEP 1: SELECT_ACTION EVALUATION ===")
print("Checking action scores for sample contexts under Fold 0 vs Fold 1:")
sample_ctxs = [
    ("Step 0, High CMI (0.6, 1.4, 0.05)", np.array([0.6, 1.4, 0.05, 0, 0, 0, 0], dtype=float)),
    ("Step 0, Low CMI  (0.1, 0.3, 0.05)", np.array([0.1, 0.3, 0.05, 0, 0, 0, 0], dtype=float)),
    ("Step 1 after LAG (0.6, 1.4, 0.50)", np.array([0.6, 1.4, 0.50, 1, 1, 0, 0], dtype=float)),
    ("Step 2 after CAEP(0.6, 1.4, 0.05)", np.array([0.6, 1.4, 0.05, 2, 1, 1, 0], dtype=float)),
]

for label, ctx in sample_ctxs:
    print(f"\nContext: {label}")
    for fname, ctrl in [("Fold 0", ctrl0), ("Fold 1", ctrl1)]:
        scores = {}
        for a in ctrl.actions:
            theta = np.linalg.inv(ctrl.A[a]) @ ctrl.b[a]
            scores[a] = float(theta @ ctx)
        chosen = ctrl.select_action(ctx)
        print(f"  {fname}: scores={scores} -> chosen={chosen}")
