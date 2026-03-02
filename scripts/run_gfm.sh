#!/usr/bin/env bash
# scripts/run_gfm.sh
set -e

# ------------------------------------------------------------
# Always prefer local repo basicts over site-packages basicts
# ------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# -----------------------
# User-configurable vars
# -----------------------
DATASETS="${DATASETS:-PEMS04,PEMS07,PEMS-BAY}"
TOKEN_K="${TOKEN_K:-64}"
CANON_M="${CANON_M:-12}"
T="${T:-12}"
H="${H:-12}"
SCALE="${SCALE:-M}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MIX_ITERS_PER_EPOCH="${MIX_ITERS_PER_EPOCH:-2000}"
MIX_PROBS="${MIX_PROBS:-uniform}"   # uniform or size
LR="${LR:-0.002}"
WD="${WD:-0.0001}"

# default: scheme B uses no external long prior at train/infer
USE_PRIOR="${USE_PRIOR:-0}"

# model cfg entry point
CFG="${CFG:-examples/cfg_tkde.py}"

# ------------------------------------------------------------
# Helper: run a command with echo
# ------------------------------------------------------------
run() { echo -e "\n\033[1;34m[RUN]\033[0m $*"; eval "$*"; }

# ------------------------------------------------------------
# Step 1: Ensure graph_state_Uk.npy exists for each dataset
# We trigger cold-start compute via runner helper by doing a tiny "dry attach".
# (If you already have graph_state_Uk.npy, this step is fast.)
# ------------------------------------------------------------
run "python - <<'PY'
import os, sys
root='${REPO_ROOT}'
if root not in sys.path: sys.path.insert(0, root)
from basicts.runners.base_tsf_runner import load_or_compute_graph_state
datasets='${DATASETS}'.split(',')
k=int('${TOKEN_K}')
for d in datasets:
    d=d.strip()
    if not d: continue
    Uk, eig = load_or_compute_graph_state(d, k, logger=None)
    print(f'[OK] {d}: U_k shape={Uk.shape}, eig shape={None if eig is None else eig.shape}')
PY"

# ------------------------------------------------------------
# Step 2: Build reference calibration (Scheme B)
# Produces:
#   datasets/canon_ref.npz
#   datasets/<G>/canon_M.npz
#   datasets/<G>/canon_ab.npz
# ------------------------------------------------------------
run "python scripts/canon/build_reference.py \
  --datasets ${DATASETS} \
  --T ${T} --H ${H} --m ${CANON_M} --k ${TOKEN_K} \
  --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} \
  --max_batches 400 --sinkhorn_iters 80 --compute_drift"

# ------------------------------------------------------------
# Step 3: Mixed pretraining (interleaved across datasets)
# This uses your existing node-domain model (KASA_TKDE) by default.
# If you want canonical model pretrain, set USE_CANON=1 below.
# ------------------------------------------------------------
RUN_TAG_PRE="mix_pre_k${TOKEN_K}_${SCALE}"
run "MULTI_DATASETS=${DATASETS} \
MIX_PROBS=${MIX_PROBS} MIX_ITERS_PER_EPOCH=${MIX_ITERS_PER_EPOCH} \
USE_PRIOR=${USE_PRIOR} \
DATASET_NAME=PEMS04 MODE=train RUN_TAG=${RUN_TAG_PRE} TOKEN_K=${TOKEN_K} SCALE=${SCALE} \
NUM_EPOCHS=${NUM_EPOCHS} BATCH_SIZE=${BATCH_SIZE} NUM_WORKERS=${NUM_WORKERS} LR=${LR} WD=${WD} \
python examples/run.py --cfg ${CFG}"

# By convention your best ckpt is saved under checkpoints/<...>/best_val*.pt
# You can set CKPT manually after it finishes:
echo -e "\n\033[1;33m[NOTE]\033[0m After pretraining, set CKPT to the best model path, e.g.:"
echo "CKPT=checkpoints/KASA_TKDE_PEMS04_${RUN_TAG_PRE}_${NUM_EPOCHS}/<hash>/KASA_TKDE_best_val_MAE.pt"

# ------------------------------------------------------------
# Step 4: LOGO Zero-shot evaluation (no finetune)
# You must set CKPT to the pretrained checkpoint path.
# ------------------------------------------------------------
cat <<'TXT'

[HOW TO RUN LOGO]
1) Export CKPT to point to your pretrained checkpoint:
   export CKPT=checkpoints/.../KASA_TKDE_best_val_MAE.pt

2) Then run zero-shot tests on target graphs:
TXT

for G in PEMS07 PEMS-BAY; do
cat <<TXT
   DATASET_NAME=${G} MODE=test RUN_TAG=zs_from_pre \
   TOKEN_K=${TOKEN_K} SCALE=${SCALE} RESUME_FROM=\$CKPT SAVE_PRED=1 \
   python examples/run.py --cfg ${CFG}
TXT
done

# ------------------------------------------------------------
# Step 5: Few-shot finetune (optional PEFT)
# FEWSHOT_RATIO: 0.01 / 0.05 / 0.1 etc.
# ------------------------------------------------------------
cat <<'TXT'

[HOW TO RUN FEW-SHOT FINETUNE]
Example (1% data, PEFT on):
   export CKPT=checkpoints/.../KASA_TKDE_best_val_MAE.pt
   DATASET_NAME=PEMS07 MODE=finetune RUN_TAG=fs1pct \
   FEWSHOT_RATIO=0.01 PEFT=1 RESUME_FROM=$CKPT \
   TOKEN_K=64 SCALE=M NUM_EPOCHS=30 \
   python examples/run.py --cfg examples/cfg_tkde.py
TXT

# ------------------------------------------------------------
# Step 6: Full finetune (baseline)
# ------------------------------------------------------------
cat <<'TXT'

[HOW TO RUN FULL FINETUNE]
   export CKPT=checkpoints/.../KASA_TKDE_best_val_MAE.pt
   DATASET_NAME=PEMS07 MODE=finetune RUN_TAG=full_ft \
   FEWSHOT_RATIO=1.0 PEFT=0 RESUME_FROM=$CKPT \
   TOKEN_K=64 SCALE=M NUM_EPOCHS=50 \
   python examples/run.py --cfg examples/cfg_tkde.py
TXT

# ------------------------------------------------------------
# Step 7: Diagnostics (you will plot drift vs conflict vs horizon)
# We already saved predictions via SAVE_PRED=1 in zero-shot.
# ------------------------------------------------------------
cat <<'TXT'

[DIAGNOSTICS]
- predictions/*.npz saved with meta json.
- Use canon_drift_js.npy + prediction errors to plot:
  (a) Drift vs Long-horizon degradation
  (b) Drift vs Gradient conflict spikes (optional: log per-batch cosine)
TXT

echo -e "\n\033[1;32mDone.\033[0m"