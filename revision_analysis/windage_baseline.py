"""Reviewer major comment #3: analytical windage baseline.

The surrogate is trained with the exact per-particle windage coefficient C_w as an input
feature, and the ground-truth simulator applies a KNOWN linear law V_windage = C_w * U10.
The reviewer's point: the 0.997 macro skill may just mean the network learned to apply a
formula it was handed, not that it discovered anything about transport dynamics. The correct
control is a baseline that applies that exact formula analytically (no learning at all) and
see how close IT gets to the same skill -- if the analytical baseline is nearly as good as the
network, the network's contribution is much smaller than the headline number suggests. If the
network clearly beats the analytical baseline, that shows it is doing extra useful work
(shear correction, interpolation effects) beyond just implementing the windage law.

Predicted residual under the analytical baseline: (du, dv) = C_w * (u10, v10)
(the same functional form the simulator itself uses to generate the ground truth).

Run: python windage_baseline.py
"""
import numpy as np

DATA_PROCESSED = r"E:\oceanography\data\processed"


def skill(pred, true):
    mse_model = ((pred - true) ** 2).sum(1).mean()
    mse_base = (true ** 2).sum(1).mean()
    return 1.0 - mse_model / mse_base, np.sqrt(mse_model), np.sqrt(mse_base)


def bootstrap_ci(pred, true, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = skill(pred[idx], true[idx])[0]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def main():
    d = np.load(f"{DATA_PROCESSED}/residual_dataset_test.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    feats = list(d["features"])
    u10_i, v10_i, wind_i, macro_i = (feats.index(k) for k in ("u10", "v10", "windage", "is_macro"))

    macro_mask = X[:, macro_i] == 1

    # Analytical windage-only prediction: (du, dv) = C_w * (u10, v10)
    pred_wind = np.column_stack([X[:, wind_i] * X[:, u10_i], X[:, wind_i] * X[:, v10_i]])

    print("=== Analytical windage-law baseline vs zero-residual baseline vs neural surrogate ===")
    print("(surrogate numbers reproduced from the independent test-set run for direct comparison)\n")
    surrogate_skill = {"macro": 0.997, "micro": 0.679, "overall": 0.943}
    for name, m in [("macro", macro_mask), ("micro", ~macro_mask), ("overall", np.ones(len(X), bool))]:
        s, rmse_m, rmse_b = skill(pred_wind[m], y[m])
        lo, hi = bootstrap_ci(pred_wind[m], y[m])
        print(f"  {name:8s}: analytical-windage skill = {s:+.3f}  [95% CI {lo:.3f}, {hi:.3f}]   "
              f"RMSE={rmse_m:.3f} m/s (baseline RMSE={rmse_b:.3f})   |  neural surrogate skill = {surrogate_skill[name]:.3f}")

    print("\nInterpretation:")
    print("  If analytical-windage skill ~= surrogate skill for macro, the surrogate's macro result")
    print("  is mostly 'correctly applying the supplied formula', not learned transport dynamics.")
    print("  The GAP between analytical and surrogate skill is the surrogate's genuine added value")
    print("  (shear correction, interpolation smoothing, current-field interaction).")


if __name__ == "__main__":
    main()
