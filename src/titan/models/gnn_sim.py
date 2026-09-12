"""Milestone 5b (Route A) — GNN learned-simulator vs MLP, on held-out particles.

Both models predict a particle's residual velocity. The MLP sees only the particle's own
cell (the Paper 1 model); the GNN attends over the 3x3 ocean neighbourhood (message passing
from ocean nodes to the particle node — the blueprint's ocean->particle attention).

Rigorous evaluation: trained on the training particles, evaluated on the INDEPENDENT held-out
particle set (different random seed) — no particle overlap, so no leakage. Repeated over
several seeds; we report mean +/- std of the residual skill score, split by debris class.

Sanity check built in: the MLP (centre-cell only) must reproduce Paper 1's ~0.94 skill, which
confirms the neighbourhood pipeline is correct before we trust the GNN comparison.

Run:
    python -m titan.models.gnn_sim
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from titan.config import DATA_PROCESSED


class MLPBaseline(nn.Module):
    """Pointwise: centre cell (7) + particle (4) -> residual. Equivalent to Paper 1."""
    def __init__(self, oc=7, pf=4, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(oc + pf, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, neigh, pfeat):
        return self.net(torch.cat([neigh[:, 4, :], pfeat], dim=1))


class GNNSimulator(nn.Module):
    """Graph-attention message passing: particle node attends over its 9 ocean-cell nodes.

    Equivalent to one GATv2-style ocean->particle layer on a star graph, batched for speed.
    """
    def __init__(self, oc=7, pf=4, h=128, heads=4):
        super().__init__()
        self.h, self.heads, self.dh = h, heads, h // heads
        self.cell = nn.Linear(oc, h)
        self.part = nn.Linear(pf, h)
        self.q = nn.Linear(h, h)
        self.k = nn.Linear(h, h)
        self.v = nn.Linear(h, h)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, neigh, pfeat):
        B = neigh.shape[0]
        ce = F.elu(self.cell(neigh))          # (B,9,h)
        pe = F.elu(self.part(pfeat))          # (B,h)
        q = self.q(pe).view(B, self.heads, self.dh)              # (B,H,dh)
        k = self.k(ce).view(B, 9, self.heads, self.dh)           # (B,9,H,dh)
        v = self.v(ce).view(B, 9, self.heads, self.dh)
        att = torch.einsum("bhd,bnhd->bnh", q, k) / (self.dh ** 0.5)  # (B,9,H)
        att = torch.softmax(att, dim=1)
        agg = torch.einsum("bnh,bnhd->bhd", att, v).reshape(B, self.h)  # (B,h)
        return self.head(torch.cat([pe, agg], dim=1))


def _skill(pred, y):
    return 1.0 - ((pred - y) ** 2).sum(1).mean() / (y ** 2).sum(1).mean()


def _load(name):
    d = np.load(DATA_PROCESSED / name, allow_pickle=True)
    return d["neigh"].astype(np.float32), d["pfeat"].astype(np.float32), d["y"].astype(np.float32), d["macro"]


def train_one(Model, tr, te, mu_o, sd_o, mu_p, sd_p, dev, seed, epochs=12, batch=8192):
    torch.manual_seed(seed)
    nt, pt, yt, _ = tr
    nv, pv, yv, macro_v = te

    def std(neigh, pf):
        return (neigh - mu_o) / sd_o, (pf - mu_p) / sd_p

    ntr, ptr = std(nt, pt)
    nva, pva = std(nv, pv)
    dl = DataLoader(TensorDataset(torch.tensor(ntr), torch.tensor(ptr), torch.tensor(yt)),
                    batch_size=batch, shuffle=True)
    model = Model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for nb, pb, yb in dl:
            opt.zero_grad()
            loss = F.mse_loss(model(nb.to(dev), pb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(nva).to(dev), torch.tensor(pva).to(dev)).cpu()
    yv_t = torch.tensor(yv)
    res = {}
    for name, m in [("micro", macro_v == 0), ("macro", macro_v == 1), ("overall", np.ones_like(macro_v, bool))]:
        res[name] = float(_skill(pred[m], yv_t[m]))
    return res


def main(seeds=(0, 1, 2)):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = _load("graphsim_train.npz")
    te = _load("graphsim_test.npz")
    print(f"[gnn-sim] train={tr[0].shape[0]} test={te[0].shape[0]} device={dev}")

    # standardisation from training data (ocean per-feature over all 9 cells; particle per-feature)
    mu_o = tr[0].reshape(-1, tr[0].shape[2]).mean(0); sd_o = tr[0].reshape(-1, tr[0].shape[2]).std(0) + 1e-6
    mu_p = tr[1].mean(0); sd_p = tr[1].std(0) + 1e-6

    out = {"MLP (centre cell)": [], "GNN (3x3 attention)": []}
    for s in seeds:
        r_mlp = train_one(MLPBaseline, tr, te, mu_o, sd_o, mu_p, sd_p, dev, s)
        r_gnn = train_one(GNNSimulator, tr, te, mu_o, sd_o, mu_p, sd_p, dev, s)
        out["MLP (centre cell)"].append(r_mlp)
        out["GNN (3x3 attention)"].append(r_gnn)
        print(f"  seed {s}: MLP overall={r_mlp['overall']:.3f}  GNN overall={r_gnn['overall']:.3f}")

    print("\n=== ROUTE A RESULT — residual skill on HELD-OUT particles (mean +/- std) ===")
    for model in out:
        for cls in ("micro", "macro", "overall"):
            arr = np.array([r[cls] for r in out[model]])
            print(f"  {model:22s} {cls:8s}  skill = {arr.mean():+.3f} +/- {arr.std():.3f}")
    mlp_o = np.array([r["overall"] for r in out["MLP (centre cell)"]]).mean()
    gnn_o = np.array([r["overall"] for r in out["GNN (3x3 attention)"]]).mean()
    print(f"\n  MLP overall {mlp_o:.3f}  (sanity: should be ~0.94, matching Paper 1)")
    print(f"  GNN overall {gnn_o:.3f}  -> neighbourhood helps by {gnn_o - mlp_o:+.3f}")
    return out


if __name__ == "__main__":
    main()
