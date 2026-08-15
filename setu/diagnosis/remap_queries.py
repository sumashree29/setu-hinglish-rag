"""
One-time migration: remap queries.json from old atomic doc_ids (D01-D20)
to the new merged corpus chunk_ids (C01-C04).
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

OLD_TO_NEW = {
    "D01": "C01", "D02": "C01", "D03": "C01", "D04": "C01", "D05": "C01",
    "D06": "C02", "D07": "C02", "D08": "C02", "D09": "C02", "D10": "C02",
    "D11": "C03", "D12": "C03", "D13": "C03", "D14": "C03",
    "D15": "C04", "D16": "C04", "D17": "C04", "D18": "C04", "D19": "C04", "D20": "C04",
}

queries = json.load(open(config.DATA_PILOT / "queries.json", encoding="utf-8"))

for q in queries:
    old_ids = q["relevant_doc_ids"]
    new_ids = sorted(set(OLD_TO_NEW[d] for d in old_ids))
    q["relevant_doc_ids"] = new_ids

out_path = Path("data/processed/queries_remapped.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)

print(f"Remapped {len(queries)} queries, saved to {out_path}")

# Quick sanity check: show first 8
for q in queries[:8]:
    print(q["query_id"], "->", q["relevant_doc_ids"])