"""Bootstrap 95% CIs for the neural surrogate's skill score on the independent test set
(reviewer minor comment #2), computed directly from the saved model checkpoint -- no
retraining, so these are exactly the headline numbers with uncertainty attached.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.models.surrogate import ResidualMLP

DATA_PROCESSED = r"E:\oceanography\data\processed"


def skill(pred, true):
    mse_model = ((pred - true) ** 2).sum(1).mean()
    mse_base = (true ** 2).sum(1).mean()
    return 1.0 - mse_model / mse_base


def bootstrap_ci(pred, true, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = skill(pred[idx], true[idx])
    return np.percentile(vals, [2.5, 97.5])


def main():
    ck = torch.load(f"{DATA_PROCESSED}/surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)

    d = np.load(f"{DATA_PROCESSED}/residual_dataset_test.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    macro_i = feats.index("is_macro")
    macro_mask = X[:, macro_i] == 1

    with torch.no_grad():
        pred = model(torch.tensor((X - mu) / sd)).numpy()

    print("=== Bootstrap 95% CI, neural surrogate, independent test set (n_boot=2000) ===")
    for name, m in [("macro", macro_mask), ("micro", ~macro_mask), ("overall", np.ones(len(X), bool))]:
        s = skill(pred[m], y[m])
        lo, hi = bootstrap_ci(pred[m], y[m])
        rmse = np.sqrt(((pred[m] - y[m]) ** 2).sum(1).mean())
        print(f"  {name:8s}: skill={s:.3f}  [95% CI {lo:.3f}, {hi:.3f}]   n={m.sum()}   RMSE={rmse:.3f} m/s")


if __name__ == "__main__":
    main()
