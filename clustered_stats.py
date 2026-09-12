"""Reviewer major comments #1-#2 (round 2): trajectory-clustered bootstrap CIs and paired
surrogate-vs-analytical-windage comparisons, replacing the anti-conservative particle-step
bootstrap used previously. Resampling unit = whole trajectory (particle), not step, since
steps within a trajectory share release parameters, forcing history, and survival selection.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, r"E:\oceanography\src")
from titan.models.surrogate import ResidualMLP

DATA_PROCESSED = r"E:\oceanography\data\processed"


def skill_from_sums(sse_model, sse_base):
    return 1.0 - sse_model / sse_base


def per_trajectory_sse(pred, true, group):
    """Sum of squared error per trajectory (2 components), and count, keyed by group id."""
    uniq, inv = np.unique(group, return_inverse=True)
    sq = ((pred - true) ** 2).sum(1)  # per-step squared error
    sse = np.bincount(inv, weights=sq, minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq))
    return uniq, sse, n


def cluster_bootstrap_skill(pred, true, group, n_boot=2000, seed=0):
    """Resample whole trajectories (with replacement), recompute pooled skill each time."""
    uniq, sse_model, n_model = per_trajectory_sse(pred, true, group)
    _, sse_base_true, _ = per_trajectory_sse(np.zeros_like(true), true, group)
    rng = np.random.default_rng(seed)
    ntraj = len(uniq)
    point = skill_from_sums(sse_model.sum(), sse_base_true.sum())
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, ntraj, ntraj)
        vals[b] = skill_from_sums(sse_model[idx].sum(), sse_base_true[idx].sum())
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, ntraj


def paired_cluster_bootstrap_rmse_diff(pred_a, pred_b, true, group, n_boot=2000, seed=0):
    """Paired difference in per-trajectory RMSE (a - b), trajectory-clustered bootstrap."""
    uniq, sse_a, n_a = per_trajectory_sse(pred_a, true, group)
    _, sse_b, n_b = per_trajectory_sse(pred_b, true, group)
    rng = np.random.default_rng(seed)
    ntraj = len(uniq)

    def pooled_rmse_diff(idx):
        rmse_a = np.sqrt(sse_a[idx].sum() / (2 * n_a[idx].sum()))
        rmse_b = np.sqrt(sse_b[idx].sum() / (2 * n_b[idx].sum()))
        return rmse_a - rmse_b

    point = pooled_rmse_diff(np.arange(ntraj))
    vals = np.array([pooled_rmse_diff(rng.integers(0, ntraj, ntraj)) for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, lo, hi, ntraj


def main():
    ck = torch.load(f"{DATA_PROCESSED}/surrogate_mlp.pt", weights_only=False)
    feats = list(ck["features"])
    model = ResidualMLP(len(feats)); model.load_state_dict(ck["state_dict"]); model.eval()
    mu, sd = ck["mu"].astype(np.float32), ck["sd"].astype(np.float32)
    macro_i = feats.index("is_macro")
    u10_i, v10_i, wind_i = feats.index("u10"), feats.index("v10"), feats.index("windage")

    d = np.load(f"{DATA_PROCESSED}/residual_dataset_test.npz", allow_pickle=True)
    X, y, group = d["X"], d["y"], d["group"]
    macro_mask = X[:, macro_i] == 1

    with torch.no_grad():
        pred_net = model(torch.tensor((X - mu) / sd)).numpy()
    pred_wind = np.column_stack([X[:, wind_i] * X[:, u10_i], X[:, wind_i] * X[:, v10_i]])

    print("=== Trajectory-clustered bootstrap: surrogate skill (independent test set) ===")
    print("(resampling unit = whole trajectory, not step; n_boot=2000)\n")
    for name, m in [("macro", macro_mask), ("micro", ~macro_mask)]:
        pt, lo, hi, ntraj = cluster_bootstrap_skill(pred_net[m], y[m], group[m])
        print(f"  {name:6s}  surrogate skill={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n_trajectories={ntraj})")

    print("\n=== Trajectory-clustered bootstrap: analytical-windage skill ===\n")
    for name, m in [("macro", macro_mask), ("micro", ~macro_mask)]:
        pt, lo, hi, ntraj = cluster_bootstrap_skill(pred_wind[m], y[m], group[m])
        print(f"  {name:6s}  windage skill={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (n_trajectories={ntraj})")

    print("\n=== Paired, trajectory-clustered RMSE difference: surrogate MINUS analytical windage ===")
    print("(negative = surrogate has lower RMSE than the analytical baseline)\n")
    for name, m in [("macro", macro_mask), ("micro", ~macro_mask)]:
        pt, lo, hi, ntraj = paired_cluster_bootstrap_rmse_diff(pred_net[m], pred_wind[m], y[m], group[m])
        sig = "excludes zero" if (lo > 0 or hi < 0) else "includes zero (not distinguishable at 95%)"
        print(f"  {name:6s}  RMSE diff={pt:+.4f} m/s  95% CI [{lo:+.4f}, {hi:+.4f}]  ({sig}, n_trajectories={ntraj})")


if __name__ == "__main__":
    main()
