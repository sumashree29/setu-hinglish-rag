# SETU-Hinglish-RAG Data Provenance & Dataset Audit

## 1. Primary In-Domain Dataset (`RBI / PM-KISAN`)
- **Corpus Chunks:** `380` canonical English chunks in `data/processed/corpus_chunks_v2.jsonl` spanning **18 categories** (`17` official RBI FAQ/public-policy topics + `1` curated pilot collection; mean length `110.61` tokens, median `84.5`, range `[17, 304]`).
- **Queries:** `314` canonical queries in `data/processed/queries_v3_final.json` (CMI mean `0.4296`, median `0.4444`, range `[0.0000, 0.6429]`; CMI bands: `14` low, `28` medium, `237` high, `35` very high).
- **Relevance Mapping:** `306` queries map to `1` gold chunk and `8` queries map to `2` gold chunks (mean `1.0255` relevant chunks per query).

## 2. Primary Query Subsets & Human Review Status
- **Pilot Subset (`Q01–Q75`):** `75` queries across `5` controlled variants (`15 en`, `15 low`, `15 medium`, `15 high`, `15 misspelled` at `Q61–Q75`).
- **Auto-Generated Paraphrases:** `239` structurally diverse `hi-en` queries (`auto_generated_v3_paraphrased`).
- **Duplicate & Leakage Grouping:** Four exact-text query pairs (`Q088/Q089`, `Q148/Q185`, `Q151/Q186`, `Q222/Q223`) and all shared gold-chunk components are strictly co-grouped into connected components before 5-fold `GroupKFold` cross-validation (verified by `tests/test_leakage.py`).
- **Human Review Limitation:** Annotation batches `data/review/review_batch_{1,2,3}.json` remain unfilled (`unreviewed`); relevance labels derive from source-document alignments rather than external human pool judgments.

## 3. Out-of-Domain Datasets (Independent & Non-Overlapping)
- **LQP Training Parallel Pairs:** `500` Hinglish–English sentence pairs from `LingoIITGN/PHINC` (zero overlap with RBI/PM-KISAN or Dolly).
- **External Validation Benchmark (`Phase 3`):** `500` duplicate-free Romanized Hinglish QA pairs (`codemix_instruction` paired with `en_input`/`en_output` passages across `open_qa`, `closed_qa`, `information_extraction`) imported from `aaditya/databricks-dolly-15k-Hinglish-Codemix` into `data/research_dolly_hinglish/` via `scripts/import_research_dataset.py`.
