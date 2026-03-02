#!/usr/bin/env bash
# scripts/run_logo_matrix.sh
set -euo pipefail

# ------------------------------------------------------------
# Always prefer local repo basicts over site-packages basicts
# ------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# -----------------------
# Required: pretrained ckpt
# -----------------------
: "${CKPT:?Please export CKPT=/path/to/pretrained_best_model.pt}"

# -----------------------
# Configurable settings
# -----------------------
CFG="${CFG:-examples/cfg_tkde.py}"

# Source graph (only for naming; not used by code)
SRC="${SRC:-PEMS04}"

# Targets to evaluate (LOGO: you can include all except SRC)
TARGETS_CSV="${TARGETS_CSV:-PEMS07,PEMS-BAY}"

# Model settings
TOKEN_K="${TOKEN_K:-64}"
SCALE="${SCALE:-M}"
USE_PRIOR="${USE_PRIOR:-0}"      # for node-domain model; keep 0 for route-B
USE_CANON="${USE_CANON:-0}"      # if you want canonical model evaluation, set 1

# Finetune epochs (few-shot typically short)
FT_EPOCHS="${FT_EPOCHS:-30}"
FULL_FT_EPOCHS="${FULL_FT_EPOCHS:-50}"

# PEFT toggle list
PEFT_LIST="${PEFT_LIST:-0,1}"    # 0=full finetune, 1=PEFT

# Shot list (0 means zero-shot test only)
SHOTS_LIST="${SHOTS_LIST:-0,1,5,10,100}"

# Default mapping from "shot" -> FEWSHOT_RATIO (override if you want)
# You can override by exporting:
#   RATIO_1=0.001  RATIO_5=0.005  RATIO_10=0.01  RATIO_100=0.1
RATIO_1="${RATIO_1:-0.001}"
RATIO_5="${RATIO_5:-0.005}"
RATIO_10="${RATIO_10:-0.01}"
RATIO_100="${RATIO_100:-0.1}"

# Prediction saving
PRED_DIR="${PRED_DIR:-predictions/logo}"
mkdir -p "${PRED_DIR}"

run() { echo -e "\n\033[1;34m[RUN]\033[0m $*"; eval "$*"; }

# Convert CSV to array
IFS=',' read -r -a TARGETS <<< "${TARGETS_CSV}"
IFS=',' read -r -a PEFTS <<< "${PEFT_LIST}"
IFS=',' read -r -a SHOTS <<< "${SHOTS_LIST}"

# Helper: shot -> ratio
shot_to_ratio() {
  local s="$1"
  case "$s" in
    1)   echo "${RATIO_1}" ;;
    5)   echo "${RATIO_5}" ;;
    10)  echo "${RATIO_10}" ;;
    100) echo "${RATIO_100}" ;;
    *)   echo "1.0" ;;
  esac
}

echo "[INFO] CKPT=${CKPT}"
echo "[INFO] SRC=${SRC}"
echo "[INFO] TARGETS=${TARGETS_CSV}"
echo "[INFO] SHOTS=${SHOTS_LIST} (ratios: 1=${RATIO_1},5=${RATIO_5},10=${RATIO_10},100=${RATIO_100})"
echo "[INFO] PEFT_LIST=${PEFT_LIST}"
echo "[INFO] USE_CANON=${USE_CANON}  USE_PRIOR=${USE_PRIOR}  SCALE=${SCALE}  TOKEN_K=${TOKEN_K}"
echo "[INFO] PRED_DIR=${PRED_DIR}"

# ------------------------------------------------------------
# 0-shot: MODE=test (no training)
# ------------------------------------------------------------
for tgt in "${TARGETS[@]}"; do
  tgt="$(echo "$tgt" | xargs)"
  [[ -z "${tgt}" ]] && continue

  tag="logo_zs_${SRC}_to_${tgt}_k${TOKEN_K}_${SCALE}"
  run "DATASET_NAME=${tgt} MODE=test RUN_TAG=${tag} \
TOKEN_K=${TOKEN_K} SCALE=${SCALE} USE_PRIOR=${USE_PRIOR} USE_CANON=${USE_CANON} \
RESUME_FROM=${CKPT} SAVE_PRED=1 PRED_SAVE_DIR=${PRED_DIR} \
python examples/run.py --cfg ${CFG}"
done

# ------------------------------------------------------------
# Few-shot / Full-shot: MODE=finetune
# ------------------------------------------------------------
for tgt in "${TARGETS[@]}"; do
  tgt="$(echo "$tgt" | xargs)"
  [[ -z "${tgt}" ]] && continue

  for shot in "${SHOTS[@]}"; do
    shot="$(echo "$shot" | xargs)"
    [[ -z "${shot}" ]] && continue

    if [[ "${shot}" == "0" ]]; then
      continue
    fi

    ratio="$(shot_to_ratio "${shot}")"

    for peft in "${PEFTS[@]}"; do
      peft="$(echo "$peft" | xargs)"
      [[ -z "${peft}" ]] && continue

      # choose epochs: for "100-shot" treat as closer to full-ft if you like
      ep="${FT_EPOCHS}"
      if [[ "${shot}" == "100" && "${ratio}" == "1.0" ]]; then
        ep="${FULL_FT_EPOCHS}"
      fi

      tag="logo_${shot}shot_${SRC}_to_${tgt}_peft${peft}_r${ratio}_k${TOKEN_K}_${SCALE}"

      run "DATASET_NAME=${tgt} MODE=finetune RUN_TAG=${tag} \
FEWSHOT_RATIO=${ratio} PEFT=${peft} RESUME_FROM=${CKPT} \
TOKEN_K=${TOKEN_K} SCALE=${SCALE} USE_PRIOR=${USE_PRIOR} USE_CANON=${USE_CANON} \
NUM_EPOCHS=${ep} SAVE_PRED=1 PRED_SAVE_DIR=${PRED_DIR} \
python examples/run.py --cfg ${CFG}"
    done
  done
done

# ------------------------------------------------------------
# Summarize prediction files into a TSV (quick scan)
# ------------------------------------------------------------
run "python - <<'PY'
import os, json, glob
import numpy as np

pred_dir = os.environ.get('PRED_DIR', 'predictions/logo')
files = sorted(glob.glob(os.path.join(pred_dir, '*.npz')))
print('[SUMMARY] files:', len(files))
print('file\\tdataset\\trun_tag\\tmode\\tfewshot_ratio\\tscaler_func\\tmae_overall')
for fp in files:
    obj = np.load(fp, allow_pickle=True)
    y_pred = obj['y_pred']; y_true = obj['y_true']
    meta = json.loads(obj['meta'].item() if hasattr(obj['meta'],'item') else str(obj['meta']))
    # y_*: [B,H,N,1]
    yp = y_pred[...,0] if y_pred.ndim==4 else y_pred
    yt = y_true[...,0] if y_true.ndim==4 else y_true
    mae = float(np.mean(np.abs(yp-yt)))
    print(f\"{os.path.basename(fp)}\\t{meta.get('dataset')}\\t{meta.get('run_tag')}\\t{meta.get('mode')}\\t{meta.get('fewshot_ratio')}\\t{meta.get('scaler_func')}\\t{mae:.4f}\")
PY"

echo -e "\n\033[1;32mDone.\033[0m  Predictions saved in ${PRED_DIR}"
echo "Next: python scripts/plot_drift_diagnostics.py --pred_dir ${PRED_DIR} --out_dir diagnostics/logo"