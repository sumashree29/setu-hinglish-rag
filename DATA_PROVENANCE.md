# Data Provenance

| Source Name | URL | Date Accessed | License | What was extracted | Preprocessing applied | Processing Script |
| --- | --- | --- | --- | --- | --- | --- |
| RBI BSBDA FAQ | https://m.rbi.org.in/scripts/FAQView.aspx?Id=144 | 2024-03-01 | licence not confirmed | FAQ pairs on Basic Savings Bank Deposit Account (BSBDA) | Chunked by QA pair, cleaned HTML, translated to Hinglish | `scripts/build_pilot_corpus.py` |
| PM-KISAN (myScheme) | https://www.myscheme.gov.in/schemes/pmkisan | 2024-03-10 | licence not confirmed | Scheme eligibility, benefits, application process | Chunked by section, cleaned HTML, translated to Hinglish | `scripts/scale_corpus_v2.py` |
| PHINC (LingoIITGN) | https://huggingface.co/datasets/LingoIITGN/PHINC | 2024-04-15 | cc-by-nc-sa-4.0 | English-Hinglish parallel sentence pairs | Filtered first 500 pairs | `setu/operators/lqp.py` |
| HinGE | https://github.com/sedilab/HinGE | 2024-04-20 | cc-by-4.0 | Hinglish generation/paraphrasing examples | Synthesized pseudo-queries for scaled retrieval | `scripts/generate_queries_v3.py` |

*Note: Dates accessed are approximate based on project history.*
