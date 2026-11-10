# Reproducibility Environment

## Verified Execution Hardware
- **CPU:** 13th Gen Intel(R) Core(TM) i7-13620H (10 physical cores / 16 logical threads)
- **RAM:** 16.0 GB (15.64 GB usable)
- **GPU Installed:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM; not used in these runs — all experiments executed deterministically on CPU for uniform latency comparisons)

## Verified Software Stack (`.venv`)
- **OS:** Windows 11 (Build `10.0.26300-SP0`, with Windows Smart App Control policy active)
- **Python Version:** Python `3.11.9`
- **Key Pinned Libraries (verified clean import under Smart App Control):**
  - `torch==2.10.0` (CPU wheel)
  - `transformers==4.57.6`, `sentence-transformers==3.3.1`
  - `numpy==2.2.0`, `scipy==1.15.0`, `pandas==3.0.3`, `scikit-learn==1.8.0`
  - `lightgbm==4.7.0`, `statsmodels==0.15.0`, `datasets==3.6.0`
  - `pyarrow==19.0.1`, `faiss-cpu==1.10.0`, `llvmlite==0.44.0`, `numba==0.61.2`, `cbor2==5.6.5`, `ranx==0.3.21`
  - `matplotlib==3.11.1`, `seaborn==0.13.2`, `rapidfuzz==3.14.5`, `pytest==9.2.0`, `psutil==7.2.2`
- **Memory Note:** Large 560M encoders (`BGE-M3`, `mE5-large`) are loaded sequentially one backbone at a time with explicit cleanup (`del`, `gc.collect()`) to fit comfortably inside 16 GB system RAM.

## Random Seed
All stochastic operations initialize `setu.config.set_seed(42)` across `random`, `numpy`, and `torch`.
