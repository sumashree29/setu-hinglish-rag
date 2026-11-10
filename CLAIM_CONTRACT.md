# Claim Contract

This document strictly defines what claims are permitted in the IEEE paper based *only* on verified GroupKFold, selective-strategy, and external Dolly-Hinglish canonical outputs.

## Safe Claims (Supported by Evidence)
- **Baseline Zero-Shot Strength:** Web-scale multilingual dense encoders (`mE5-large` 0.8767 in-domain / 0.8695 external; `BGE-M3` 0.8526 in-domain / 0.9121 external) resolve 77.4–88.6% of tested Hinglish queries at rank 1 without query modification.
- **Overcorrection Hazard from Standalone Operators:** Unconditionally applying standalone corrective operators (`LAG`, `CAEP`, `LQP` depending on model/domain) causes statistically significant MRR degradation on initially correct queries across both the 314-query RBI corpus (`-0.0310` to `-0.0757`, Holm $p \le 0.0220$) and the 500-query external Dolly-Hinglish corpus (`-0.0182` to `-0.0874`, Holm $p \le 0.0145$).
- **Controller Step & Latency Reduction (H8):** Under leak-free 5-fold GroupKFold CV, `SETU_v2` reduces mean non-STOP actions by 54.25% (`3.000 -> 1.373`, Holm $p = 1.79 \times 10^{-55}$) and mean CPU latency by 59.1% (`1186.80 ms -> 485.10 ms`) relative to `SETU_v1`.
- **Selective Recovery Without Overcorrection (`SETU_v2` & `mE5+LQP`):** In-domain GroupKFold `SETU_v2` avoids significant damage on working queries (`-0.0075`, Holm $p = 0.9225$) while delivering significant recovery on initially failed queries (`+0.0364`, Holm $p = 0.0494$). Out-of-domain on Dolly-Hinglish, `PHINC`-trained `LQP` lifts failed queries on `mE5-large` (`+0.0934`, Holm $p = 0.0008$).
- **Confidence Correlation (H10):** Initial score margin (`confidence_trace[0]`) correlates positively with MRR (`Spearman rho = 0.5030`, raw $p = 1.53 \times 10^{-21}$, Holm $p = 1.07 \times 10^{-20}$).
- **Empirical Oracle Bound:** An omniscient per-query selector (`pi*`) reaches `0.8842` MRR on `BGE-M3` (`+0.0316` over RAW, Holm $p = 5 \times 10^{-6}$), selecting unmodified `RAW` on `281/314` (89.49%) queries.

## Conditional Claims (Require Precise Scientific Wording)
- **Controller Sequence Diversity & CMI Association:** Report exact sequence counts (`LQP->LAG->STOP`: 188, `STOP`: 71, `LQP->STOP`: 55), `75.8%` `repeat_action_forced` terminations, raw Chi-square `p = 0.0043` vs. 17-test Holm `p_corr = 0.0731`, and H9 step-vs-CMI `rho = 0.0053 (p = 0.9255)`. Never claim complete context independence.
- **Overall Pipeline Parity (H6):** State accurately that no statistically significant overall MRR difference was detected between `SETU_v1` (0.8492) and `SETU_v2` (0.8550) (`p_corr = 0.8334`). This is not an equivalence claim.

## Prohibited Claims (Do NOT Write These)
- Do not claim overall `SETU_v1` or `SETU_v2` achieves statistically significant full-corpus MRR superiority over `RAW` (`p_corr >= 0.8334`).
- Do not claim confidence gating *causes* retrieval gains from correlation metrics alone.
- Do not claim standalone `CAEP`/`LAG`/`LQP` reliably recover failed queries in-domain (`p_corr >= 0.3300`).
- Do not fabricate human review completion (`data/review/review_batch_{1,2,3}.json` remain unreviewed).
