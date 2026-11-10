import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    errors = []

    try:
        tables = read_json(ROOT / "results" / "tables" / "ieee_ready_tables.json")
    except Exception as e:
        print(f"FAILED TO LOAD IEEE TABLES: {e}")
        sys.exit(1)

    canonical_raw_mrr = f"{tables['Table_2_Baseline']['mrr']:.4f}"

    # 1. Documents to check
    doc_paths = [
        "RESULTS_SUMMARY.md",
        "README.md",
        "CLAIM_CONTRACT.md",
        "CLAIM_EVIDENCE_MATRIX.md",
        "FINAL_EVIDENCE_REPORT.md",
        "FINAL_EVIDENCE_SCOPE.md",
    ]

    for name in doc_paths:
        try:
            content = read_file(ROOT / name)
        except Exception:
            errors.append(f"File {name} not found.")
            continue

        if "TBD" in content:
            errors.append(f"{name} contains 'TBD'")

        lower_content = content.lower()
        if "statistically equivalent" in lower_content or "statistical equivalence" in lower_content:
            errors.append(f"{name} claims 'statistically equivalent' without TOST/equivalence margin.")

        if name == "RESULTS_SUMMARY.md" and canonical_raw_mrr not in content:
            errors.append(f"{name} is missing the exact canonical RAW MRR ({canonical_raw_mrr})")

        if "proves causality" in lower_content or "causes retrieval" in lower_content:
            errors.append(f"{name} contains unsupported causal claims.")

        if "completely independent" in lower_content:
            errors.append(f"{name} contains exaggerated 'completely independent' controller claims.")

    # 2. Check main.tex against canonical tables
    tex_path = ROOT / "ieee_submission" / "main.tex"
    if tex_path.exists():
        tex = read_file(tex_path)
        if "26 distinct categories" in tex:
            errors.append("main.tex still claims '26 distinct categories' (actual corpus has 18 categories).")
        if "indic-sentence-bert-nli" in tex:
            errors.append("main.tex contains wrong model name 'indic-sentence-bert-nli'.")
        if canonical_raw_mrr not in tex:
            errors.append(f"main.tex is missing canonical RAW MRR ({canonical_raw_mrr}).")

        # Check Table 3 canonical values
        t3 = tables.get("Table_3_RAW_v1_v2", {})
        for sys_name in ["SETU_v1", "SETU_v2"]:
            if sys_name in t3:
                mrr_str = f"{t3[sys_name]['mrr']:.4f}"
                if mrr_str not in tex:
                    errors.append(f"main.tex is missing {sys_name} canonical MRR ({mrr_str}).")

        # Check Table 6 Holm p-values for BGE-M3 LAG
        t6 = tables.get("Table_6_Overcorrection", {})
        bge_lag_p = t6.get("bge_m3", {}).get("operators", {}).get("LAG", {}).get("wilcoxon_p_correct_holm")
        if bge_lag_p is not None and f"{bge_lag_p:.4f}" not in tex:
            errors.append(f"main.tex is missing canonical BGE-M3 LAG p_corr ({bge_lag_p:.4f}).")
    else:
        errors.append("ieee_submission/main.tex not found.")

    # 3. Check references.bib hygiene
    bib_path = ROOT / "ieee_submission" / "references.bib"
    if bib_path.exists():
        bib = read_file(bib_path)
        if "and others" in bib:
            errors.append("references.bib still contains incomplete 'and others' author placeholder.")
    else:
        errors.append("ieee_submission/references.bib not found.")

    if errors:
        print("CONSISTENCY ERRORS FOUND:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)

    print("ALL CONSISTENCY CHECKS PASSED.")
    sys.exit(0)


if __name__ == "__main__":
    main()
