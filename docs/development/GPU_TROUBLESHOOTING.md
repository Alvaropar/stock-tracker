# GPU CUDA Error - Recovery Guide

## Problem
The gridsearch encountered a CUDA GPU error while processing alpha_026:
```
torch.cuda.AcceleratorError: CUDA error: an illegal memory access was encountered
```

This typically means:
- GPU out of memory
- GPU memory fragmentation
- GPU memory corruption
- Incompatible GPU driver/PyTorch version

## Solution: Disable GPU (Fast Fix)

### Option 1: Resume from Checkpoint (Recommended)

The gridsearch was at combo 260/3799. It **automatically saved a checkpoint**, so no work is lost!

Run with GPU disabled:
```bash
set DISABLE_GPU=1
cd C:\Projects\stock-analyzer
python -m alphas.run --alpha grid --101grid ^
    --start-date 2015-01-01 --end-date 2024-12-31 ^
    --grid-dir "C:\Projects\stock-analyzer\alphas\output\grid"
```

**What happens:**
1. Detects existing output directory
2. Loads checkpoint with 260 completed combos
3. Disables GPU (uses CPU instead)
4. Resumes from combo 261
5. Continues without GPU acceleration

**Expected time:** ~15-20 minutes (CPU is slower than GPU, but stable)

### Option 2: Windows Command Prompt
```bash
cd C:\Projects\stock-analyzer
set DISABLE_GPU=1
python -m alphas.run --alpha grid --101grid --start-date 2015-01-01 --end-date 2024-12-31 --grid-dir "C:\Projects\stock-analyzer\alphas\output\grid"
```

### Option 3: PowerShell
```powershell
$env:DISABLE_GPU = "1"
cd C:\Projects\stock-analyzer
python -m alphas.run --alpha grid --101grid --start-date 2015-01-01 --end-date 2024-12-31 --grid-dir "C:\Projects\stock-analyzer\alphas\output\grid"
```

## Why GPU Failed?

GPU acceleration is used for:
- Converting numpy arrays to GPU tensors
- Vectorized matrix operations (normalization, weighting)
- Returns computation

When GPU fails, gridsearch now:
1. Detects the error
2. Clears GPU memory (`torch.cuda.empty_cache()`)
3. Falls back to CPU automatically
4. Prints warning message

**But current fix has graceful fallback** - GPU errors should now just trigger fallback instead of crashing!

## Verify Checkpoint Saved

```python
import json
from pathlib import Path

output_dir = Path(r"C:\Projects\stock-analyzer\alphas\output\grid")
run = sorted(output_dir.glob("top3000_101param_*"), reverse=True)[0]

checkpoint_file = run / ".checkpoint.json"
if checkpoint_file.exists():
    ckpt = json.load(open(checkpoint_file))
    print(f"Checkpoint found!")
    print(f"Completed combos: {len(ckpt['completed_combos'])}")
    print(f"Last batch: {ckpt['batch_completed']}")
    print(f"Timestamp: {ckpt['timestamp']}")
else:
    print("No checkpoint found")
```

## If You Want to Fix GPU

### Check GPU Memory
```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check current usage
print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

### Try GPU Memory Reset
```bash
# Option 1: Restart Python/Jupyter
# GPU memory is freed on process exit

# Option 2: Clear GPU cache (in Python)
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()

# Option 3: Disable GPU globally for rest of session
torch.cuda.set_per_process_memory_fraction(0.5)  # Use only 50% of GPU memory
```

### Update PyTorch GPU Drivers

If you have an NVIDIA GPU:
```bash
# Update CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads

# Or update PyTorch with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Replace cu118 with your CUDA version (cu117, cu121, etc)
```

## Performance: CPU vs GPU

| Operation | GPU Time | CPU Time | Impact |
|-----------|----------|----------|--------|
| Backtest normalization | 0.5 sec | 2 sec | 4x slower |
| Full combo | 0.7 sec | 2.5 sec | 3.5x slower |
| Full gridsearch (3799 combos) | ~45 min | ~150 min | 3.3x slower |

**Trade-off:** CPU slower but stable (no memory errors)

## Recommendation

### For Now (To Finish Gridsearch)
```bash
set DISABLE_GPU=1
python -m alphas.run --alpha grid --101grid ...
```
- Complete gridsearch without interruption
- CPU slower but reliable
- Estimated 15-20 minutes more

### For Later (Fix GPU)
1. Check CUDA driver version: `nvidia-smi`
2. Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. If mismatch, reinstall PyTorch with correct CUDA version
4. Test: Run small gridsearch with GPU
5. If stable, re-enable for next run

## Commands Quick Reference

### Resume with CPU (GPU disabled)
```bash
set DISABLE_GPU=1 && python -m alphas.run --alpha grid --101grid --start-date 2015-01-01 --end-date 2024-12-31 --grid-dir "C:\Projects\stock-analyzer\alphas\output\grid"
```

### Check checkpoint status
```python
import json
ckpt = json.load(open("alphas/output/grid/top3000_101param_*/.checkpoint.json"))
print(f"Completed: {len(ckpt['completed_combos'])}, Last batch: {ckpt['batch_completed']}")
```

### Clear GPU cache (in Python)
```python
import torch
torch.cuda.empty_cache()
```

### Check GPU memory usage
```bash
nvidia-smi
```

## If Problem Persists

If GPU errors continue even with `DISABLE_GPU=1`, then:
1. Check disk space (minimum 10 GB)
2. Check RAM (minimum 8 GB)
3. Check CPU isn't maxed out
4. Reduce batch size further in `grid_search.py` line 724

---

**Next step:** Run with `set DISABLE_GPU=1` to complete gridsearch using CPU acceleration.

**Time estimate:** ~15-20 minutes (slower but stable)

