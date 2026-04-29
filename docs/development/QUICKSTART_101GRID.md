# Quick Start: Running --101grid Parametrized Alpha Gridsearch

## Prerequisites
- [x] All fixes applied (operators.py, grid_search.py)
- [x] D: drive with 10+ GB free space
- [x] Python environment with numpy, pandas, torch
- [x] Integration tests passing ✓

## 1. Run the Gridsearch

```bash
cd C:\Projects\stock-analyzer

python -m alphas.run \
    --alpha grid \
    --101grid \
    --start-date 2015-01-01 \
    --end-date 2024-12-31 \
    --grid-dir "C:\Projects\stock-analyzer\alphas\output\grid"
```

**Expected output:**
```
[101grid] Operator caching enabled (disk: D:/operator_cache)
[101grid] Loading alphas_parametizable module...
[101grid] Building combinations (101 alphas × parameter sets)...
[101grid] Processing 10,100 combinations in batches of 8...
[101grid] Batch 1/1263: Computing 8 signals...
[101grid] Batch 1/1263: Normalizing batch (zscore)...
[101grid] Batch 1/1263: Backtesting in parallel (4 workers)...
...
[101grid] Operator caching disabled, disk cache persisted
```

## 2. Monitor Progress

### Watch Cache Growing
```bash
# In a separate terminal
cd /d D:\operator_cache
dir /s

# Rerun every minute to see growth
# Should reach ~2-4 GB by end of gridsearch
```

### Check Output Directory
```bash
dir "C:\Projects\stock-analyzer\alphas\output\grid\top3000_101param_YYYY-MM-DD\"
# Should show:
# - results.csv (10,100 rows)
# - ranked_alphas.csv (top performers)
# - metadata.json (run config)
# - config.json (settings)
# - errors.log (any warnings)
```

## 3. After Gridsearch Completes

### Review Results
```bash
cd C:\Projects\stock-analyzer\alphas\output\grid\top3000_101param_YYYY-MM-DD

# Top 10 alphas by OOS Sharpe
head -11 ranked_alphas.csv | column -t -s','
```

### Analyze Full Results
```python
import pandas as pd

results = pd.read_csv('results.csv')
print(results[['alpha', 'params', 'sharpe_oos', 'mean_ic_oos']].head(10))

# By alpha (collapse all parameter sets)
by_alpha = results.groupby('alpha').agg({
    'sharpe_oos': 'max',
    'mean_ic_oos': 'max'
}).sort_values('sharpe_oos', ascending=False)
print(by_alpha.head(15))
```

## 4. Cache Management

### Check Cache Status
```bash
# Windows
dir D:\operator_cache /s | find "File(s)"

# Should show ~500-1000 .pkl files
# Total size should be ~2-4 GB
```

### Reuse Cache for Next Run
The cache persists automatically. Next gridsearch will:
1. Skip computing operators already cached
2. Use L1 (memory) cache for frequent operations
3. Load from L2 (disk) cache if L1 miss

This gives 2-3x speedup on subsequent runs with overlapping parameters.

### Clear Cache Manually (Optional)
Only do this after major code changes to operators:
```bash
rmdir /s /q D:\operator_cache
# Cache will be recreated on next --101grid run
```

## 5. Timing Expectations

### First Run
- **Total time:** 10-15 minutes (with optimizations)
- **Disk I/O:** 5-10 min (cache writes)
- **Compute:** 5 min (parallel backtest)

### Subsequent Runs (Cache Warm)
- **Total time:** 6-10 minutes
- **Cache hits:** ~80% on overlapping parameters
- **New compute:** Only new parameter combinations

### Scaling
- **101 alphas** ✓ Optimal
- **Per alpha:** ~100 parameter combinations
- **Total combos:** ~10,100
- **Per combo:** 70ms backtest (IS + OOS)

## 6. Troubleshooting

### Error: "D: drive not available"
```python
# Option 1: Use C: temp directory
python -m alphas.run --alpha grid --101grid --start-date 2015-01-01 --end-date 2024-12-31

# Option 2: Edit grid_search.py line 602 to use different drive
_ops_mod.enable_operator_cache(cache_dir="E:/operator_cache")
```

### Error: "Out of memory"
- Reduce batch size in grid_search.py (line ~640): `BATCH_SIZE = 4` (instead of 8)
- Or: Reduce cache by clearing: `rmdir /s /q D:\operator_cache`

### Warning: "Operator caching could not enable"
- Check D: drive is accessible: `dir D:\`
- Check disk space: `dir D:\ | find "free"`
- Logs will show fallback to in-memory caching (still works, just slower)

### Slow Progress (No GPU)
- CPU-only is fine, just slower
- If you have CUDA: Install torch[cuda]: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Grid will auto-detect GPU and accelerate normalization

## 7. Advanced Options

### Use Different Cache Drive
Edit `C:\Projects\stock-analyzer\alphas\grid_search.py` line 602:
```python
_ops_mod.enable_operator_cache(cache_dir="F:/fast_ssd")  # Use F: instead
```

### Use RAM Disk (Advanced)
1. Install WinRAMDisk or similar
2. Create 8-16 GB RAM disk at R:\
3. Edit grid_search.py:
```python
_ops_mod.enable_operator_cache(cache_dir="R:/operator_cache")
```
**Advantage:** Fastest possible cache
**Disadvantage:** Cache lost on reboot

### Adjust Parallel Workers
Edit `C:\Projects\stock-analyzer\alphas\grid_search.py` around line 680:
```python
with ThreadPoolExecutor(max_workers=6) as executor:  # Changed from 4
```
Recommend: `CPU_COUNT - 2` for full utilization

## 8. Next Steps

After gridsearch completes:
1. Review `ranked_alphas.csv` for best parameter sets
2. Use top-performing alphas in portfolio construction
3. Run `--alpha portfolio` with selected alphas
4. Keep cache for iterative refinement

---

**Status:** ✓ Infrastructure Ready | ✓ Cache System Working | ✓ Tests Passing

Ready to execute: `python -m alphas.run --alpha grid --101grid ...`
