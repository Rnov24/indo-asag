# Cross-check findings and fixes for Exp 5, 6, 7, 8

## Exp 5 Issues:
# 1. ❌ Line 79: loads `_emb_` files which are now gitignored. Will fail on fresh Colab clone
# 2. ❌ Line 138: push cell not hidden (@title missing)  
# 3. ❌ Line 175: still has `checkpoints/` in git add (unnecessary for Exp 5)
# 4. ❌ No `subprocess` import at top (only imported in push cell)
# FIX: Exp 5 DEPENDS on emb files, so those need to be available. Either:
#   a) Generate emb in Exp 3 and push to GitHub (too large), or
#   b) Re-generate emb in Exp 5, or  
#   c) Upload emb to HF Hub and download in Exp 5

## Exp 6 Issues:
# 1. ❌ Line 70: header still says "Utilitas Push Per-Checkpoint"
# 2. ❌ Line 110: comment "setiap seed langsung di-push ke GitHub" (stale)
# 3. ❌ Line 197: `results/models/` in git add — Ridge .pkl files are small, OK to keep
# 4. ✅ Rest is clean

## Exp 7 Issues:
# 1. ❌ Line 82: unused `_repo_url = None`
# 2. ❌ Line 162: orphaned `np.save(loqo_hc_preds)` in wrong section (between HC and IndoBERT)
# 3. ❌ Line 170: stale comment about "Push otomatis setelah setiap soal"  
# 4. ✅ Final push is clean

## Exp 8 Issues:
# 1. ❌ Line 193: push cell not hidden (@title missing)
# 2. ❌ Line 230: still has `checkpoints/` in git add
# 3. ❌ No `os.makedirs(METRICS_DIR)` — will crash if metrics dir doesn't exist
# 4. ✅ Logic and statistics are correct
