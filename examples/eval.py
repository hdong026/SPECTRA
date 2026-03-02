import os, json, glob
import numpy as np

def mae(a,b): return np.mean(np.abs(a-b))
def rmse(a,b): return np.sqrt(np.mean((a-b)**2))

def load_Uk(dataset_dir, k=64):
    path = os.path.join(dataset_dir, "graph_state_Uk.npy")
    Uk = np.load(path).astype(np.float32)[:, :k]  # [N,k]
    return Uk

def drift_curve(y_pred, y_true, Uk):
    # y_*: [B,H,N,1]
    r = (y_pred - y_true)[..., 0]  # [B,H,N]
    B,H,N = r.shape
    UkT = Uk.T  # [k,N]

    # energy distribution over k structural modes
    p = []
    for h in range(H):
        # [B,N] -> coeff [B,k]
        coeff = r[:, h, :] @ Uk  # (B,N)(N,k)=(B,k)
        e = np.mean(coeff**2, axis=0) + 1e-12  # [k]
        p_h = e / np.sum(e)
        p.append(p_h)
    p0 = p[0]
    drift = [0.5*np.sum(np.abs(ph - p0)) for ph in p]  # L1/2
    auc = float(np.trapz(drift, dx=1.0) / max(H-1,1))
    return drift, auc

def main(pred_dir, token_k=64):
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    for f in files:
        z = np.load(f, allow_pickle=True)
        y_pred = z["y_pred"]; y_true = z["y_true"]
        meta = json.loads(str(z["meta"]))
        dataset = meta["dataset"]
        H = y_pred.shape[1]

        # horizon metrics
        maes = [mae(y_pred[:,h], y_true[:,h]) for h in range(H)]
        rmses = [rmse(y_pred[:,h], y_true[:,h]) for h in range(H)]

        # drift
        Uk = load_Uk(os.path.join("datasets", dataset), k=token_k)
        drift, auc = drift_curve(y_pred, y_true, Uk)

        print("="*80)
        print(os.path.basename(f), "dataset=", dataset)
        print("MAE@1/3/6/12:", maes[0], maes[2], maes[5], maes[11] if H>=12 else maes[-1])
        print("RMSE@1/3/6/12:", rmses[0], rmses[2], rmses[5], rmses[11] if H>=12 else rmses[-1])
        print("drift AUCΔ:", auc)
        print("drift:", drift[:min(12,H)])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, default="predictions")
    ap.add_argument("--token_k", type=int, default=int(os.getenv("TOKEN_K","64")))
    args = ap.parse_args()
    main(args.pred_dir, token_k=args.token_k)