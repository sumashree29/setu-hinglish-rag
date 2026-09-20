# Reproducibility Environment

## Hardware
- **CPU**: AMD Ryzen 5 5600H with Radeon Graphics
- **Cores**: 6 cores, 12 threads
- **RAM**: 16 GB
- **GPU Used**: No (All experiments run exclusively on CPU)

## Software
- **OS**: Windows 11 Home Single Language (Version 10.0.22631 Build 22631)
- **Python Version**: Python 3.11.x
- **Environment Variables**:
  - `OMP_NUM_THREADS` / `MKL_NUM_THREADS`: Not explicitly set (defaults used).

## Random Seed
All operations are deterministic and initialized by `setu.config.set_seed(42)` which fixes random states for `random`, `numpy`, and `torch`.
