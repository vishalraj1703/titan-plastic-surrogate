"""Reviewer major comment #8 (round 2): the network is supplied the EXACT per-particle
windage coefficient, unavailable in most real forecasting settings. This ablation asks how
much skill depends on that exact value, by substituting the class-mean C_w (the only
information realistically available without particle-level metadata) at inference time,
using the existing trained checkpoint -- no retraining.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.models.surrogate import ResidualMLP

DATA_PROCESSED = r"E:\oceanography\data\processed"


def skill(pred, true):
    return 1.0 - ((pred - true) ** 2).sum(1).mean() / (true ** 2).sum(1).mean()


def main():
    ck = torch.load(f"{DATA_PROCESSED}/surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats)); model.load_state_dict(ck["state_dict"]); model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)
    macro_i, wind_i = feats.index("is_macro"), feats.index("windage")

    d = np.load(f"{DATA_PROCESSED}/residual_dataset_test.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    macro_mask = X[:, macro_i] == 1

    print("=== Windage-coefficient availability ablation (independent test set) ===")
    print("Exact Cw (as trained) vs. class-mean Cw substituted at inference (no retraining)\n")
    for name, m in [("macro", macro_mask), ("micro", ~macro_mask)]:
        Xm = X[m].copy()
        with torch.no_grad():
            pred_exact = model(torch.tensor((Xm - mu) / sd)).numpy()
        s_exact = skill(pred_exact, y[m])

        Xm_mean = Xm.copy()
        Xm_mean[:, wind_i] = Xm[:, wind_i].mean()  # class-mean Cw, exact value withheld
        with torch.no_grad():
            pred_mean = model(torch.tensor((Xm_mean - mu) / sd)).numpy()
        s_mean = skill(pred_mean, y[m])

        print(f"  {name:6s}  skill with EXACT per-particle Cw = {s_exact:.3f}   "
              f"skill with only CLASS-MEAN Cw = {s_mean:.3f}   (drop = {s_exact - s_mean:.3f})")


if __name__ == "__main__":
    main()
