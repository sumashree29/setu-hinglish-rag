# PAPER NOTES — Framing the SETU Results for IEEE Submission

## What the Evidence Actually Supports

After rigorous Phase 1–5 pilot/domain-scale evaluation on 314 queries and 380 corpus chunks across 3 embedding models, the evidence paints a clear picture that differs from the original hypotheses but is equally publishable:

### The Original Hypothesis (what we expected)
Code-mixed Hinglish queries degrade dense retrieval quality, and SETU's correction operators (LQP, CAEP, LAG) recover that degradation.

### What We Found Instead

1. **Code-mixing does NOT reliably degrade retrieval** (H1 not supported, ρ=0.09, p=0.11). BGE-M3 handles code-mixed queries about as well as monolingual ones on this domain corpus. The presumption of degradation — the entire motivation for building correction operators — is not supported by our data.

2. **Indic-tuned models are dramatically WORSE, not better** (H2 significant in opposite direction, p=1e-16). Indic-SBERT (MRR=0.604) underperforms BGE-M3 (MRR=0.847) by a massive margin. The "Indic-tuned models handle code-mixing better" assumption is wrong for our domain.

3. **Correction operators hurt aggregate performance** (H4 not supported, H7 significant in opposite direction). When applied unconditionally, LQP/CAEP/LAG each reduce MRR because they over-correct the 76% of queries where the base model already succeeds.

4. **BUT operators massively help the 24% of failing queries** (overcorrection diagnosis). On queries where RAW fails (MRR<1.0), operators boost MRR by +0.48 on average. The problem is not the operators themselves — it's applying them indiscriminately.

5. **The learned controller solves this** (H8 supported, p≈0). SETU v2 learns to STOP immediately for easy queries and only invokes operators when they're likely to help. This achieves v1-equivalent quality with 70% fewer steps and 40% lower latency.

6. **The controller's efficiency win is NOT sophisticated sequencing** — it learned a simpler insight: most queries don't need correction. The "smart" thing is doing nothing most of the time.

---

## Recommended Paper Framing

### Title Options
- "When Not to Correct: Adaptive Gating for Code-Mixed Query Correction in Dense Retrieval"
- "SETU: Why Code-Mixed Correction Operators Need Learned Gating, Not Better Algorithms"
- "Negative Results and Adaptive Recovery: Testing the Code-Mixed Retrieval Degradation Hypothesis"

### Abstract Skeleton

> We rigorously test the assumption that Hinglish code-mixing degrades dense retrieval quality in a domain-specific RAG system. Using a 380-chunk RBI banking FAQ corpus and 314 code-mixed queries across three embedding models, we find **no evidence of systematic CMI-driven degradation** (H1: ρ=0.09, p=0.11). Three purpose-built correction operators (LQP, CAEP, LAG) — designed to recover degradation via embedding projection, entity augmentation, and adaptive rewriting — each **hurt aggregate retrieval** when applied unconditionally, because they over-correct the 76% of queries the base model already handles correctly.
>
> However, a stratified analysis reveals that operators provide a **+0.48 MRR boost** on the 24% of queries where the base model fails. We introduce a contextual bandit controller (LinUCB) that learns to selectively gate operator application, achieving matched retrieval quality with 70% fewer pipeline steps (1.2 vs 4.0 mean steps, p≈0). Our findings suggest that for code-mixed retrieval, **adaptive operator selection** is more important than operator design — the key challenge is knowing *when* to intervene, not *how*.

### Key Contributions (in order of strength)
1. **Over-correction diagnosis**: Formal demonstration that correction operators help failing queries (+0.48 MRR) but harm successful ones (-0.16 MRR), with the net effect depending on the base model's accuracy distribution.
2. **Adaptive gating**: A LinUCB controller that learns to gate operators, resolving the over-correction problem and achieving 70% step reduction at matched quality.
3. **Negative result with diagnostic value**: Code-mixing alone does not degrade BGE-M3 retrieval on a controlled FAQ corpus, contradicting the assumption underlying prior code-mixed-RAG correction work.
4. **Methodological contribution**: Open-source pipeline for diagnosis, operator training, and adaptive evaluation on code-mixed retrieval.

### Results Sections to Write
1. **§4.1 Degradation Hypothesis** — H1 (null), H2 (opposite direction)
2. **§4.2 Operator Effectiveness** — H4 (null), H7 (opposite direction), over-correction diagnosis
3. **§4.3 Adaptive Controller** — H6 (equivalent quality), H8 (fewer steps), H9 (null CMI-steps correlation)
4. **§4.4 Confidence Calibration** — H10 (null at scale, null within each band)
5. **§4.5 Limitations**: 
    - **Construct Validity of CMI**: The LID tagger relies on a hand-rolled lexicon, making CMI scores potentially noisy.
    - **Pilot-Scale Corpus**: The corpus is limited to 380 chunks, restricting generalizability.
    - **LAG In-sample Labeling**: LAG's training labels were derived from in-sample trajectory optimization rather than a strict hold-out fold.
    - **Missing MIRACL Benchmark**: Evaluated only on the domain-specific corpus; MIRACL public benchmark arm was not completed due to data-loading issues.
    - **CMI Band Imbalance**: Severe skew toward high CMI.

### What NOT to Claim
- ❌ Do not claim SETU "corrects code-mixed retrieval degradation" — H1 shows degradation isn't reliably present
- ❌ Do not claim operators improve retrieval — they hurt aggregate MRR (H4, H7)
- ❌ Do not claim the controller found smart operator sequences — it learned to STOP
- ❌ Do not present citation-only baselines (mSPLADE etc.) as if they were measured on our corpus
- ❌ Do not claim confidence proxy is useful for gating — H10 is null

### What TO Claim
- ✅ The degradation hypothesis does not hold for modern multilingual models (BGE-M3, mE5-large) on domain-specific corpora
- ✅ Correction operators have real conditional value (conditional on base model failure)
- ✅ Adaptive gating via bandit control is essential and yields measurable efficiency gains
- ✅ This is a rigorous negative-result-with-diagnostic-value paper for the RAG-correction literature
