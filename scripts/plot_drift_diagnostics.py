# scripts/plot_drift_diagnostics.py
from __future__ import annotations

import os
import json
import argparse
from glob import glob
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def _safe_json_load(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def _horizon_mae(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    y_pred/y_true: [B,H,N,1] or [B,H,N]
    Return: mae_by_h [H]
    """
    if y_pred.ndim == 4:
        y_pred = y_pred[..., 0]
    if y_true.ndim == 4:
        y_true = y_true[..., 0]
    assert y_pred.shape == y_true.shape, (y_pred.shape, y_true.shape)
    # abs -> mean over B,N
    mae_h = np.mean(np.abs(y_pred - y_true), axis=(0, 2))
    return mae_h


def _degradation_metrics(mae_h: np.ndarray) -> Tuple[float, float]:
    """
    Given mae_h [H], return:
      - auc: area under curve (mean over horizons)
      - slope: linear fit slope over horizon index
    """
    H = mae_h.shape[0]
    x = np.arange(1, H + 1, dtype=np.float32)
    auc = float(np.mean(mae_h))
    # slope via least squares: slope = cov(x,y)/var(x)
    xm = x.mean()
    ym = mae_h.mean()
    slope = float(np.sum((x - xm) * (mae_h - ym)) / (np.sum((x - xm) ** 2) + 1e-12))
    return auc, slope


def _guess_source_from_ckpt(resume_from: str, dataset_list: List[str]) -> Optional[str]:
    """
    Heuristic: if resume_from path contains a dataset name, treat it as source.
    Works for ckpt paths like checkpoints/KASA_TKDE_PEMS04_xxx/...
    """
    if not resume_from:
        return None
    lower = resume_from.lower()
    for d in dataset_list:
        if d.lower() in lower:
            return d
    return None


def _load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    obj = np.load(path, allow_pickle=True)
    y_pred = obj["y_pred"]
    y_true = obj["y_true"]
    meta = _safe_json_load(obj["meta"].item() if hasattr(obj["meta"], "item") else str(obj["meta"]))
    return y_pred, y_true, meta


def _load_canon_M_ab(dataset: str, root: str = "datasets") -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    m_path = os.path.join(root, dataset, "canon_M.npz")
    ab_path = os.path.join(root, dataset, "canon_ab.npz")
    if not (os.path.exists(m_path) and os.path.exists(ab_path)):
        return None, None, None

    m = np.load(m_path, allow_pickle=True)["M"].astype(np.float32)  # [m,k]
    ab = np.load(ab_path, allow_pickle=True)
    a = ab["a"].astype(np.float32).reshape(-1)  # [m]
    b = ab["b"].astype(np.float32).reshape(-1)  # [k]
    return m, a, b


def _calibrated_M(M: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # diag(a) M diag(b) in energy space is a^2 * M * b^2 (because M=E[Z^2])
    aa = (a ** 2).reshape(-1, 1)
    bb = (b ** 2).reshape(1, -1)
    return aa * M * bb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=str, default="PEMS04,PEMS07,PEMS-BAY",
                    help="Comma-separated dataset names in the SAME order as canon_drift_js.npy.")
    ap.add_argument("--drift_path", type=str, default="datasets/canon_drift_js.npy")
    ap.add_argument("--pred_dir", type=str, default="predictions")
    ap.add_argument("--out_dir", type=str, default="diagnostics")
    ap.add_argument("--filter_tag", type=str, default="", help="Only use prediction files whose filename contains this substring.")
    ap.add_argument("--show", action="store_true", help="Show figures interactively.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    D = len(dataset_list)

    # ---- load drift matrix ----
    if not os.path.exists(args.drift_path):
        raise FileNotFoundError(f"drift file not found: {args.drift_path}")
    drift = np.load(args.drift_path).astype(np.float32)
    assert drift.shape == (D, D), f"drift shape {drift.shape} != ({D},{D})"

    name2idx = {n: i for i, n in enumerate(dataset_list)}

    # ---- scan prediction files ----
    files = sorted(glob(os.path.join(args.pred_dir, "*.npz")))
    if args.filter_tag:
        files = [f for f in files if args.filter_tag in os.path.basename(f)]
    if len(files) == 0:
        raise RuntimeError(f"No prediction npz found in {args.pred_dir} (filter_tag={args.filter_tag})")

    # Collect points: each file -> (target, source, drift, auc, slope, overall_mae)
    rows = []
    per_target_curves: Dict[str, List[np.ndarray]] = {}

    for fp in files:
        y_pred, y_true, meta = _load_npz(fp)

        target = meta.get("dataset", None) or os.path.basename(fp).split("_")[0]
        if target not in name2idx:
            # skip unknown dataset (not in drift matrix ordering)
            continue

        # compute horizon metrics on stored scale (already rescaled when saved)
        mae_h = _horizon_mae(y_pred, y_true)
        auc, slope = _degradation_metrics(mae_h)
        overall = float(np.mean(np.abs((y_pred[..., 0] if y_pred.ndim == 4 else y_pred) -
                                       (y_true[..., 0] if y_true.ndim == 4 else y_true))))

        # infer source dataset from ckpt path
        src = _guess_source_from_ckpt(meta.get("resume_from", ""), dataset_list)
        # if cannot infer, treat as "unknown" and use drift=nan
        dval = np.nan
        if src is not None and src in name2idx:
            dval = float(drift[name2idx[src], name2idx[target]])

        rows.append((fp, src, target, dval, auc, slope, overall))

        per_target_curves.setdefault(target, []).append(mae_h)

    if len(rows) == 0:
        raise RuntimeError("No usable prediction files matched datasets list. "
                           "Check --datasets order and prediction meta['dataset'].")

    # -----------------------------
    # Figure 1: Drift heatmap
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.imshow(drift, aspect="auto")
    plt.xticks(range(D), dataset_list, rotation=30, ha="right")
    plt.yticks(range(D), dataset_list)
    plt.colorbar()
    plt.title("Spectral Drift (JS) Heatmap")
    plt.tight_layout()
    f1 = os.path.join(args.out_dir, "drift_heatmap.png")
    plt.savefig(f1, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # -----------------------------
    # Figure 2: Drift vs AUC (long-horizon degradation proxy)
    # -----------------------------
    xs = np.array([r[3] for r in rows], dtype=np.float32)
    ys_auc = np.array([r[4] for r in rows], dtype=np.float32)
    mask = np.isfinite(xs)

    plt.figure(figsize=(7, 5))
    plt.scatter(xs[mask], ys_auc[mask])
    # trend line
    if mask.sum() >= 2:
        coef = np.polyfit(xs[mask], ys_auc[mask], deg=1)
        xx = np.linspace(xs[mask].min(), xs[mask].max(), 50)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy)
    plt.xlabel("Drift (JS)")
    plt.ylabel("AUC of MAE(h) (mean over horizons)")
    plt.title("Drift vs Long-Horizon Degradation (AUC)")
    plt.tight_layout()
    f2 = os.path.join(args.out_dir, "drift_vs_auc.png")
    plt.savefig(f2, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # -----------------------------
    # Figure 3: Drift vs slope (how fast MAE grows with horizon)
    # -----------------------------
    ys_slope = np.array([r[5] for r in rows], dtype=np.float32)

    plt.figure(figsize=(7, 5))
    plt.scatter(xs[mask], ys_slope[mask])
    if mask.sum() >= 2:
        coef = np.polyfit(xs[mask], ys_slope[mask], deg=1)
        xx = np.linspace(xs[mask].min(), xs[mask].max(), 50)
        yy = coef[0] * xx + coef[1]
        plt.plot(xx, yy)
    plt.xlabel("Drift (JS)")
    plt.ylabel("Slope of MAE(h)")
    plt.title("Drift vs Horizon-Error Growth (Slope)")
    plt.tight_layout()
    f3 = os.path.join(args.out_dir, "drift_vs_slope.png")
    plt.savefig(f3, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # -----------------------------
    # Figure 4: Per-horizon MAE curves by target dataset
    # -----------------------------
    plt.figure(figsize=(8, 6))
    for tgt, curves in per_target_curves.items():
        # average multiple runs for same target
        mcurve = np.mean(np.stack(curves, axis=0), axis=0)
        plt.plot(np.arange(1, mcurve.shape[0] + 1), mcurve, label=tgt)
    plt.xlabel("Horizon h")
    plt.ylabel("MAE(h)")
    plt.title("Per-Horizon MAE Curves (from saved predictions)")
    plt.legend()
    plt.tight_layout()
    f4 = os.path.join(args.out_dir, "mae_h_curves.png")
    plt.savefig(f4, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # -----------------------------
    # Figure 5: Before/After calibration energy maps (per dataset)
    # -----------------------------
    for tgt in dataset_list:
        M, a, b = _load_canon_M_ab(tgt, root="datasets")
        if M is None:
            continue
        M2 = _calibrated_M(M, a, b)

        # normalize for visualization (sum=1)
        Mv = M / (M.sum() + 1e-12)
        M2v = M2 / (M2.sum() + 1e-12)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(Mv, aspect="auto")
        plt.title(f"{tgt}: BEFORE (p_G)")
        plt.xlabel("graph freq λ-index")
        plt.ylabel("time freq ω-index")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1, 2, 2)
        plt.imshow(M2v, aspect="auto")
        plt.title(f"{tgt}: AFTER calib (p̃_G)")
        plt.xlabel("graph freq λ-index")
        plt.ylabel("time freq ω-index")
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        fout = os.path.join(args.out_dir, f"energy_before_after_{tgt}.png")
        plt.savefig(fout, dpi=200)
        if args.show:
            plt.show()
        plt.close()

    # -----------------------------
    # Save a summary table (csv-like)
    # -----------------------------
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("file\tsource\ttarget\tdrift_js\tauc_mae_h\tslope_mae_h\toverall_mae\n")
        for fp, src, tgt, dval, auc, slope, overall in rows:
            f.write(f"{os.path.basename(fp)}\t{src}\t{tgt}\t{dval}\t{auc}\t{slope}\t{overall}\n")

    print("[DONE] Saved figures to:", args.out_dir)
    print("  -", f1)
    print("  -", f2)
    print("  -", f3)
    print("  -", f4)
    print("  - energy_before_after_<dataset>.png")
    print("  - summary:", summary_path)


if __name__ == "__main__":
    main()