import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    print("Running Final Readiness Gate...")
    
    # 1. Run consistency checker
    print("1. Running Consistency Checker...")
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_paper_consistency.py")], capture_output=True, text=True)
    if res.returncode != 0:
        print("Consistency Checker FAILED:")
        print(res.stdout)
        print("IEEE_READY = FALSE")
        sys.exit(1)
        
    print("Consistency Checker PASSED.")
    
    # 2. Check canonical evidence
    print("2. Checking Canonical Evidence...")
    canonical_files = ["latency.json", "per_query_metrics.json", "setu_trajectories.json"]
    for f in canonical_files:
        if not (ROOT / "results" / "canonical" / f).exists():
            print(f"Missing canonical evidence: {f}")
            print("IEEE_READY = FALSE")
            sys.exit(1)
            
    print("Canonical Evidence PASSED.")
    
    # 3. Check IEEE Tables exist
    if not (ROOT / "results" / "tables" / "ieee_ready_tables.json").exists():
        print("Missing IEEE Tables.")
        print("IEEE_READY = FALSE")
        sys.exit(1)
        
    print("Readiness checks completed.")
    print("\nIEEE_READY = TRUE")
    sys.exit(0)

if __name__ == "__main__":
    main()
