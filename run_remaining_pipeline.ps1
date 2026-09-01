Write-Host "Running Final V1 vs V2 Comparison on Scaled Corpus..."
python -u scripts/compare_setu_v1_v2_scaled.py

Write-Host "Running Statistical Tests..."
python -u scripts/run_statistical_tests_h1_h10_scaled.py

Write-Host "Assembling Figures..."
python -u scripts/assemble_figures.py

Write-Host "ALL DONE!"
