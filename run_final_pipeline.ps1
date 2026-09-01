# Wait for task 196 (label_lag_queries) to finish if it's still running, but since we are just executing, we can just run them all in order.
# Actually, since label_lag_queries is running in the background, we should wait for it.
# We will just write a script that does the whole pipeline.

Write-Host "Running Empirical LAG Labeling..."
python -u scripts/label_lag_queries.py

Write-Host "Running LAG Training and Ablation..."
python -u scripts/train_and_ablate_operators_v2.py

Write-Host "Running Final V1 vs V2 Comparison on Scaled Corpus..."
python scripts/compare_setu_v1_v2_scaled.py

Write-Host "Running Statistical Tests..."
python scripts/run_statistical_tests_h1_h10_scaled.py

Write-Host "Assembling Figures..."
python scripts/assemble_figures.py

Write-Host "ALL DONE!"
