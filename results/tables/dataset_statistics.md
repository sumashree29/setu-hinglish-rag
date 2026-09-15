# Dataset Statistics

## Document Statistics
- **Source Documents**: 1
- **Total Chunks**: 380

### Chunk Length Distribution (Words)
- **Mean**: 110.61
- **Median**: 84.5
- **Min**: 17
- **Max**: 304

## Query Statistics
- **Total Queries**: 314

### Queries by Variant
- en: 15
- low: 15
- medium: 15
- high: 15
- misspelled: 15
- hi-en: 239

### Queries by Review Status
- unreviewed: 75
- auto_generated_v3_paraphrased: 239

### CMI Distribution
- **Mean**: 0.430
- **Median**: 0.444
- **Min**: 0.000
- **Max**: 0.643

### CMI Band Counts
- low: 14
- high: 237
- medium: 28
- very_high: 35

### LID Entropy Distribution
- **Mean**: 1.158
- **Median**: 1.231
- **Min**: 0.000
- **Max**: 1.577

### Relevance Labels
- **Mean relevant docs per query**: 1.03
- **Queries with >1 relevant doc**: 8

## Relevance-Label Construction Procedure
Each query has exactly one gold chunk inherited from its `source_question`'s parent chunk.
Labels are binary, not graded. No pooling was performed.
**Consequence:** `Recall@10 = 0.987` is near-saturation, so Recall is uninformative on this corpus and **MRR / nDCG@10 / Hit@1 must be the primary metrics**.
