"""Milestone 3 — the GO/NO-GO experiment: minimal residual-velocity surrogate.

Per the scope-discipline agreement, this is the SIMPLEST possible model (a small MLP), not
the full 5-head HeteroGAT. It answers exactly one question:

    Can a learned surrogate predict the residual velocity that OceanParcels produces,
    beating the naive "just follow the current" baseline (residual = 0)?

If yes -> the ambitious GNN is worth building and we have the core paper result.
If no  -> we learn that cheaply, now, before investing in graph attention.

Metric: skill score  S = 1 - MSE(model) / MSE(zero-residual baseline), per behavioural class.
S = 0 means "no better than pure advection"; S -> 1 means "residual fully captured".

Run (after torch is installed):
    python -m titan.models.surrogate
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from titan.config import DATA_PROCESSED


class ResidualMLP(nn.Module):
    """Maps [ocean + wind + particle features] -> residual velocity (du, dv) in m/s."""

    def __init__(self, n_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def _skill(pred, true):
    """1 - MSE(pred)/MSE(0). Baseline is predicting zero residual (pure advection)."""
    mse_model = ((pred - true) ** 2).sum(1).mean()
    mse_base = (true ** 2).sum(1).mean()
    return 1.0 - mse_model / mse_base, mse_model, mse_base


def main(epochs: int = 40, batch: int = 4096, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[surrogate] device = {dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    d = np.load(DATA_PROCESSED / "residual_dataset.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    mu, sd = d["mu"], d["sd"]
    groups = d["group"]
    feats = list(d["features"])
    macro_idx = feats.index("is_macro")
    is_macro_col = X[:, macro_idx].copy()

    Xs = (X - mu) / sd  # standardize features

    # TRAJECTORY-LEVEL split: whole particles go to train OR val, never split mid-track.
    # This prevents temporal leakage (seeing steps 1..k of a particle in train and k+1 in val).
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_val_g = len(uniq) // 5
    val_groups = set(uniq[:n_val_g].tolist())
    is_val = np.isin(groups, list(val_groups))
    val_idx = np.where(is_val)[0]
    tr_idx = np.where(~is_val)[0]
    print(f"[surrogate] trajectory-level split: {len(uniq)-n_val_g} train / {n_val_g} val particles")

    Xt = torch.tensor(Xs[tr_idx]); yt = torch.tensor(y[tr_idx])
    Xv = torch.tensor(Xs[val_idx]); yv = torch.tensor(y[val_idx])
    macro_v = is_macro_col[val_idx]

    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch, shuffle=True)

    model = ResidualMLP(Xs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xv.to(dev)).cpu()
            s_all, mm, mb = _skill(pv, yv)
            print(f"  epoch {ep+1:3d}  val skill={s_all:.3f}  MSE_model={mm:.4f}  MSE_base={mb:.4f}")

    # final per-class evaluation
    model.eval()
    with torch.no_grad():
        pv = model(Xv.to(dev)).cpu()
    print("\n=== GO/NO-GO RESULT (residual-velocity skill) ===")
    for name, m in [("micro", macro_v == 0), ("macro", macro_v == 1), ("overall", np.ones_like(macro_v, bool))]:
        s, mm, mb = _skill(pv[m], yv[m])
        rmse = np.sqrt(float(((pv[m] - yv[m]) ** 2).sum(1).mean()))
        print(f"  {name:8s}: skill={float(s):+.3f}   residual RMSE={rmse:.3f} m/s   (baseline RMSE={np.sqrt(float(mb)):.3f})")

    # Independent held-out test set (different particles AND different random seed) — the
    # strongest, leakage-free evaluation. Overrides the in-run validation numbers for the paper.
    test_path = DATA_PROCESSED / "residual_dataset_test.npz"
    if test_path.exists():
        dt = np.load(test_path, allow_pickle=True)
        Xt2, yt2 = dt["X"], dt["y"]
        macro_t = Xt2[:, macro_idx].copy()
        with torch.no_grad():
            pt = model(torch.tensor((Xt2 - mu) / sd).to(dev)).cpu()
        yt2t = torch.tensor(yt2)
        print("\n=== INDEPENDENT TEST-SET SKILL (leakage-free) ===")
        for name, m in [("micro", macro_t == 0), ("macro", macro_t == 1),
                        ("overall", np.ones_like(macro_t, bool))]:
            s, mm, mb = _skill(pt[m], yt2t[m])
            rmse = np.sqrt(float(((pt[m] - yt2t[m]) ** 2).sum(1).mean()))
            print(f"  {name:8s}: skill={float(s):+.3f}   residual RMSE={rmse:.3f} m/s   "
                  f"(baseline RMSE={np.sqrt(float(mb)):.3f})")

    out = DATA_PROCESSED / "surrogate_mlp.pt"
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd, "features": feats}, out)
    print(f"\n[surrogate] saved {out}")


if __name__ == "__main__":
    main()
