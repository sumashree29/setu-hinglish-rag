import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from setu.config import set_seed
import json
import random
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with open(root / "data" / "processed" / "queries_v3_final.json", "r", encoding="utf-8") as f:
    queries = json.load(f)

set_seed()
sample = random.sample(queries, 100)

for idx, q in enumerate(sample):
    tokens = q["text"].split()
    print(f"Q{idx}: {q['text']}")
    # print(f"    Tags: {['' for _ in tokens]}")
