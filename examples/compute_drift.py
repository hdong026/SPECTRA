import os, glob, json
import numpy as np
from scipy.signal import welch

def psd_welch(x, nperseg=288, noverlap=144):
    nperseg = min(nperseg, len(x))
    noverlap = min(noverlap, max(0, nperseg // 2 - 1))
    _, pxx = welch(x, nperseg=nperseg, noverlap=noverlap, detrend="constant")
    return pxx

def drift_auc(npz_path, channel=0, nperseg=288, noverlap=144):
    z = np.load(npz_path, allow_pickle=True)
    y_pred = z["y_pred"]  # [S,Q,N,C]
    y_true = z["y_true"]
    meta = json.loads(str(z["meta"]))

    S, Q, N, C = y_pred.shape
    drifts = []
    for h in range(Q):
        psd_p_list, psd_t_list = [], []
        for n in range(N):
            p = y_pred[:, h, n, channel]
            t = y_true[:, h, n, channel]
            psd_p_list.append(psd_welch(p, nperseg, noverlap))
            psd_t_list.append(psd_welch(t, nperseg, noverlap))
        psd_p = np.mean(np.stack(psd_p_list, axis=0), axis=0)
        psd_t = np.mean(np.stack(psd_t_list, axis=0), axis=0)

        psd_p = psd_p / (psd_p.sum() + 1e-12)
        psd_t = psd_t / (psd_t.sum() + 1e-12)
        drifts.append(np.abs(psd_p - psd_t).sum())

    drifts = np.asarray(drifts)
    return meta, drifts, float(drifts.mean())

if __name__ == "__main__":
    pred_dir = os.getenv("PRED_DIR", "predictions")
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    print("Found", len(files), "files in", pred_dir)

    for f in files:
        meta, dr, auc = drift_auc(f)
        print(meta.get("dataset"), meta.get("run_tag"), "AUCΔ", auc)