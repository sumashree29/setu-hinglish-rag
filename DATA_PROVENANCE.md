# SETU-Hinglish-RAG Data Provenance & Dataset Audit

## 1. Dataset Dimensions
- **Total Documents (Chunks)**: 380 canonical chunks in `data/processed/corpus_chunks_v2.jsonl`
- **Total Queries**: 314 canonical queries in `data/processed/queries_v3_final.json`
- **Relevance Mapping**: Strict single-positive per query. Each query in the evaluation maps exactly to one target gold document derived from its parent question object.

## 2. Query Subsets
- **Manually Verified (Pilot)**: 75 queries
- **Auto-Generated (Synthetic)**: 239 queries
- **Misspelled Entity Subset**: Q61-Q75 (15 queries) explicitly injected with misspellings/transliteration errors to stress-test the CAEP operator.

## 3. Lexical Overlap & Scale Risks
- **Overlap Ratio**: The Phase 9 audit logged a semantic overlap ratio between the auto-generated and pilot queries of 0.83x. This indicates high structural and lexical redundancy within the evaluation set.
- **Domain Scale**: Operating over just 380 domain chunks strongly biases retrieval toward trivial lexical matching rather than deep semantic reasoning. Models with strong zero-shot cross-lingual capacities (like BGE-M3 and mE5-large) easily map these 314 queries to the small 380-chunk space, achieving native MRRs above 0.85 without any operator assistance. Evaluating complex pipelines on this scale carries severe risks of ceiling effects.
