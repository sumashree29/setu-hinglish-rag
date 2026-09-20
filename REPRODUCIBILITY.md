# Reproducibility
## Seed Management
Every entrypoint script centralizes random state initialization using `from setu.config import set_seed; set_seed()`. This guarantees seed=42 across numpy, python random, and torch globally. Local scripts do not call `np.random.seed(42)` individually.
