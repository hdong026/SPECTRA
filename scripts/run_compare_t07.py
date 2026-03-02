# scripts/run_compare_t07.py

# ===== MUST be before any heavy imports to avoid OpenMP/MKL fork segfault =====
import os
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import csv
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import glob

# ===== patterns =====
VAL_RE = re.compile(r"Result <val>:\s*\[.*?val_MAE:\s*([0-9.]+)", re.IGNORECASE)
TEST_RE = re.compile(
    r"Result <test>:\s*\[.*?test_MAE:\s*([0-9.]+),\s*test_RMSE:\s*([0-9.]+),\s*test_MAPE:\s*([0-9.]+)\]",
    re.IGNORECASE
)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_last_test(output: str):
    ms = TEST_RE.findall(output)
    if not ms:
        return None
    mae, rmse, mape = ms[-1]
    return float(mae), float(rmse), float(mape)

def run_cmd(env: dict, cmd: list[str]) -> str:
    """Run command and return stdout+stderr."""
    e = os.environ.copy()
    e.update({k: str(v) for k, v in env.items() if v is not None})

    print("\n[RUN]", " ".join(cmd))
    print("[ENV]", {k: e[k] for k in sorted(env.keys())})

    p = subprocess.run(
        cmd, env=e,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    out = p.stdout
    print(out)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return out

def run_with_early_stop(env: dict, cmd: list[str], patience: int, min_delta: float):
    """
    Run training, monitor val_MAE, stop when no improvement for `patience` validations.
    Returns:
        output(str), best_val(float or inf)
    """
    e = os.environ.copy()
    e.update({k: str(v) for k, v in env.items() if v is not None})

    print("\n[RUN]", " ".join(cmd))
    print("[ENV]", {k: e[k] for k in sorted(env.keys())})

    proc = subprocess.Popen(
        cmd, env=e,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True
    )

    best = float("inf")
    bad = 0
    out_lines = []

    try:
        for line in proc.stdout:
            out_lines.append(line)
            print(line, end="")

            m = VAL_RE.search(line)
            if m:
                v = float(m.group(1))
                if v < best - min_delta:
                    best = v
                    bad = 0
                else:
                    bad += 1
                    if bad >= patience:
                        print(f"\n[EARLY-STOP] val_MAE did not improve for {patience} validations. Stop now. best={best:.4f}")
                        proc.send_signal(signal.SIGINT)
                        break

        # wait graceful exit
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()

    finally:
        if proc.stdout:
            proc.stdout.close()

    return "".join(out_lines), best

def find_best_ckpt(model_name: str, dataset: str, run_tag: str, num_epochs: int) -> str:
    """
    checkpoints/{model}_{dataset}_{run_tag}_{num_epochs}/{md5}/{model}_best_val_MAE.pt
    """
    base = Path("checkpoints") / f"{model_name}_{dataset}_{run_tag}_{num_epochs}"
    pats = [
        str(base / "*" / f"{model_name}_best_val_MAE.pt"),
        str(base / "**" / f"{model_name}_best_val_MAE.pt"),
        str(base / "**" / "*best_val_MAE*.pt"),
    ]
    files = []
    for p in pats:
        files.extend(glob.glob(p, recursive=True))
    if files:
        files.sort(key=lambda x: os.path.getmtime(x))
        return files[-1]

    # fallback: last epoch ckpt
    pats2 = [str(base / "**" / f"{model_name}_*.pt")]
    files2 = []
    for p in pats2:
        files2.extend(glob.glob(p, recursive=True))
    if not files2:
        raise FileNotFoundError(f"No ckpt found under {base}")
    files2.sort(key=lambda x: os.path.getmtime(x))
    return files2[-1]

def save_csv(rows, path):
    ensure_dir(Path(path).parent)
    cols = ["target", "ratio", "method", "MAE", "RMSE", "MAPE", "run_tag", "best_val_MAE", "best_ckpt"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print("[SAVE]", path)

def plot(rows, metric, out_path, target):
    # Delay matplotlib import to avoid fork segfault issues
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(Path(out_path).parent)
    methods = sorted(set(r["method"] for r in rows if r["target"] == target))
    plt.figure()
    for m in methods:
        pts = [(float(r["ratio"]), float(r[metric])) for r in rows if r["target"] == target and r["method"] == m]
        pts.sort(key=lambda x: x[0])
        xs = [p[0] * 100 for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", label=m)
    plt.xlabel("Label ratio (%)")
    plt.ylabel(metric)
    plt.title(f"{target} {metric} vs label ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("[PLOT]", out_path)

def train_then_eval_best(tag: str, env_train: dict, args, ratio: float, method: str, rows: list):
    """
    Train (early stop) -> find best ckpt -> eval_only test -> append to rows
    """
    env_train = dict(env_train)
    env_train["TEST_INTERVAL"] = str(args.train_test_interval)

    _, best_val = run_with_early_stop(
        env_train,
        ["python", args.train_entry, "-c", args.cfg, "--gpus", args.gpus],
        patience=args.patience,
        min_delta=args.min_delta
    )

    best_ckpt = find_best_ckpt(
        model_name=args.model_name,
        dataset=args.target,
        run_tag=tag,
        num_epochs=args.num_epochs
    )

    env_eval = {
        "MODE": "test",
        "DATASET_NAME": args.target,
        "RUN_TAG": f"eval_{tag}",
        "RESUME_FROM": best_ckpt,
        "SIGN_FLIP_TRIALS": str(args.sign_flip_trials),
    }
    if args.save_pred:
        env_eval["SAVE_PRED"] = "1"

    out_eval = run_cmd(env_eval, ["python", args.eval_entry, "-c", args.cfg])
    m = parse_last_test(out_eval)
    if m is None:
        raise RuntimeError(f"cannot parse test metrics from eval_only for {tag}")
    mae, rmse, mape = m

    rows.append({
        "target": args.target,
        "ratio": ratio,
        "method": method,
        "MAE": mae, "RMSE": rmse, "MAPE": mape,
        "run_tag": tag,
        "best_val_MAE": best_val if best_val < 1e18 else "",
        "best_ckpt": str(best_ckpt)
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="examples/cfg_tkde_es07.py")
    ap.add_argument("--train_entry", default="examples/run.py")
    ap.add_argument("--eval_entry", default="examples/eval_only.py")
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--target", default="PEMS07")
    ap.add_argument("--pretrain_ckpt", required=True)
    ap.add_argument("--ratios", default="0.01,0.05,0.10,0.20,0.50,1.0")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min_delta", type=float, default=0.0)
    ap.add_argument("--num_epochs", type=int, default=100)
    ap.add_argument("--save_pred", action="store_true")
    ap.add_argument("--sign_flip_trials", type=int, default=0)
    ap.add_argument("--train_test_interval", type=int, default=999999)
    ap.add_argument("--model_name", default="KASA_TKDE")
    args = ap.parse_args()

    ratios = [r.strip() for r in args.ratios.split(",") if r.strip()]
    ts = datetime.now().strftime("%m%d_%H%M%S")
    rows = []
    ensure_dir("results")

    for r in ratios:
        r_float = float(r)

        # scratch
        tag = f"scratch_r{r}_t07_{ts}"
        env_train = {
            "MODE": "finetune",
            "DATASET_NAME": args.target,
            "RUN_TAG": tag,
            "FEWSHOT_RATIO": r,
            "PEFT": "0",
            "NUM_EPOCHS": args.num_epochs,
        }
        if args.save_pred:
            env_train["SAVE_PRED"] = "1"
        train_then_eval_best(tag, env_train, args, r_float, "scratch", rows)

        # full finetune
        tag = f"full_r{r}_t07_{ts}"
        env_train = {
            "MODE": "finetune",
            "DATASET_NAME": args.target,
            "RUN_TAG": tag,
            "FEWSHOT_RATIO": r,
            "RESUME_FROM": args.pretrain_ckpt,
            "PEFT": "0",
            "NUM_EPOCHS": args.num_epochs,
        }
        if args.save_pred:
            env_train["SAVE_PRED"] = "1"
        train_then_eval_best(tag, env_train, args, r_float, "full", rows)

        # PEFT
        tag = f"peft_r{r}_t07_{ts}"
        env_train = {
            "MODE": "finetune",
            "DATASET_NAME": args.target,
            "RUN_TAG": tag,
            "FEWSHOT_RATIO": r,
            "RESUME_FROM": args.pretrain_ckpt,
            "PEFT": "1",
            "NUM_EPOCHS": args.num_epochs,
        }
        if args.save_pred:
            env_train["SAVE_PRED"] = "1"
        train_then_eval_best(tag, env_train, args, r_float, "peft", rows)

        csv_path = f"results/compare_t07_{ts}.csv"
        save_csv(rows, csv_path)

    csv_path = f"results/compare_t07_{ts}.csv"
    save_csv(rows, csv_path)

    plot(rows, "MAE",  f"results/curve_t07_{ts}_MAE.png",  args.target)
    plot(rows, "RMSE", f"results/curve_t07_{ts}_RMSE.png", args.target)
    plot(rows, "MAPE", f"results/curve_t07_{ts}_MAPE.png", args.target)

    print("\nDone.")
    print("CSV:", csv_path)

if __name__ == "__main__":
    main()