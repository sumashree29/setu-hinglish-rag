# Paper Notes & Narrative Constraints

This file records the exact narrative boundaries for drafting and updating the IEEE paper based on the freshly executed GroupKFold evaluation, selective-correction strategies, and external Dolly-Hinglish validation.

## 1. Scope & External Generalization
The primary RBI evaluation covers 314 Hinglish queries across 380 chunks (18 categories), paired with an out-of-domain 500-query Dolly-Hinglish QA validation benchmark. Web-scale dense baselines (`BGE-M3` 0.8526 in-domain / 0.9121 external; `mE5-large` 0.8767 in-domain / 0.8695 external) already resolve 77–89% of queries zero-shot.

## 2. Controller Adaptivity & Efficiency (H6, H8)
Under leak-free 5-fold GroupKFold CV:
- No statistically significant overall MRR difference is detected between `SETU_v1` (0.8492) and `SETU_v2` (0.8550) (`p_raw=0.2482`, `p_corr=0.8334`), and we do not claim equivalence without TOST testing.
- `SETU_v2` cuts operator actions by 54.25% (`3.000 -> 1.373`, `p_corr=1.79e-55`) and latency by 59.1% (`1186.8 ms -> 485.1 ms`).
- However, sequence diversity remains restricted to 3 macro-patterns (`LQP->LAG->STOP`: 188, `STOP`: 71, `LQP->STOP`: 55), with 75.8% `repeat_action_forced` terminations due to missing action masking and compressed score-margin features (`313/314 < 0.20`). Raw Chi-square between first action and CMI band is `0.0043` (`p_corr=0.0731` after 17-test Holm correction), while total steps show no significant correlation with CMI (H9 `rho=0.0053, p=0.9255`).

## 3. Overcorrection vs. Selective Recovery (Table IV & External Dolly)
- Standalone operators (`LQP`, `CAEP`, `LAG`) cause significant degradation on already-correct queries across both datasets (Holm `p_corr <= 0.0220`) while failing to yield Holm-significant lift on baseline failures in-domain (`p_corr >= 0.3300`).
- Selective `SETU_v2` avoids significant overcorrection (`-0.0075, p_corr=0.9225`) and achieves statistically significant lift on RBI failed queries (`+0.0364, p_corr=0.0494`), while out-of-domain zero-shot `PHINC`->`Dolly` transfer shows `mE5-large + LQP` achieving `+0.0934` (`p_corr=0.0008`) on failed queries.

## 4. Confidence Gating (H10) & Empirical Oracle (`pi*`)
Score-margin confidence strongly correlates with MRR (`rho=0.5030, p=1.53e-21, p_corr=1.07e-20`), validating margin gating diagnostics. Empirical oracle `pi*` attains MRR `0.8842` (`+0.0316`, Holm `p=5e-6`), choosing unmodified `RAW` on 281/314 (`89.49%`) queries.

## 5. Language and Tone
- Report Holm-Bonferroni corrected p-values alongside raw p-values where relevant.
- Never claim equivalence ("no statistically significant difference detected" only).
- Never claim causality from confidence correlations.
- Explicitly note unfilled human review batches (`data/review/review_batch_{1,2,3}.json`) in Limitations.
