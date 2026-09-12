"""Milestone 5b (Route A) — beaching prediction: GNN vs MLP (blueprint Head 4).

Binary task: will a particle beach within 24 h? Evaluated on the INDEPENDENT held-out particle
set, over several seeds, with ROC-AUC and average precision (robust to class balance).

  * MLP : centre cell only (land-mask there is always 0) -> blind to the coastline.
  * GNN : attends over the 7x7 neighbourhood incl. the land mask -> sees the shore.

If the GNN wins here, it is because it can perceive coastline geometry that a pointwise model
structurally cannot — a genuine, honest demonstration of what graph message passing buys.

Run:
    python -m titan.models.gnn_beach
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from titan.config import DATA_PROCESSED

CENTER = 24  # centre of a 7x7 grid (index 3*7+3)


class MLPBeach(nn.Module):
    def __init__(self, cf=5, pf=4, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cf + pf, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, neigh, pfeat):
        return self.net(torch.cat([neigh[:, CENTER, :], pfeat], dim=1)).squeeze(1)


class GNNBeach(nn.Module):
    def __init__(self, cf=5, pf=4, h=128, heads=4):
        super().__init__()
        self.h, self.heads, self.dh = h, heads, h // heads
        self.cell = nn.Linear(cf, h)
        self.part = nn.Linear(pf, h)
        self.q = nn.Linear(h, h); self.k = nn.Linear(h, h); self.v = nn.Linear(h, h)
        self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(),
                                  nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, neigh, pfeat):
        B, Ncell, _ = neigh.shape
        ce = F.elu(self.cell(neigh)); pe = F.elu(self.part(pfeat))
        q = self.q(pe).view(B, self.heads, self.dh)
        k = self.k(ce).view(B, Ncell, self.heads, self.dh)
        v = self.v(ce).view(B, Ncell, self.heads, self.dh)
        att = torch.softmax(torch.einsum("bhd,bnhd->bnh", q, k) / (self.dh ** 0.5), dim=1)
        agg = torch.einsum("bnh,bnhd->bhd", att, v).reshape(B, self.h)
        return self.head(torch.cat([pe, agg], dim=1)).squeeze(1)


def _load(name):
    d = np.load(DATA_PROCESSED / name, allow_pickle=True)
    return d["neigh"].astype(np.float32), d["pfeat"].astype(np.float32), d["y"].astype(np.float32), d["macro"]


def train_one(Model, tr, te, mu_o, sd_o, mu_p, sd_p, dev, seed, epochs=25, batch=2048):
    torch.manual_seed(seed)
    nt, pt, yt, _ = tr; nv, pv, yv, macro_v = te
    ntr = (nt - mu_o) / sd_o; ptr = (pt - mu_p) / sd_p
    nva = (nv - mu_o) / sd_o; pva = (pv - mu_p) / sd_p
    dl = DataLoader(TensorDataset(torch.tensor(ntr), torch.tensor(ptr), torch.tensor(yt)),
                    batch_size=batch, shuffle=True)
    model = Model().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for nb, pb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(nb.to(dev), pb.to(dev)), yb.to(dev))
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logit = model(torch.tensor(nva).to(dev), torch.tensor(pva).to(dev)).cpu().numpy()
    prob = 1 / (1 + np.exp(-logit))
    res = {}
    for name, m in [("micro", macro_v == 0), ("macro", macro_v == 1), ("overall", np.ones_like(macro_v, bool))]:
        if yv[m].sum() > 0 and yv[m].sum() < m.sum():
            res[name] = (roc_auc_score(yv[m], prob[m]), average_precision_score(yv[m], prob[m]))
    return res


def main(seeds=(0, 1, 2)):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = _load("beach_train.npz"); te = _load("beach_test.npz")
    print(f"[beach] train={tr[0].shape[0]} test={te[0].shape[0]} neighbourhood={tr[0].shape[1]} cells device={dev}")
    mu_o = tr[0].reshape(-1, tr[0].shape[2]).mean(0); sd_o = tr[0].reshape(-1, tr[0].shape[2]).std(0) + 1e-6
    mu_p = tr[1].mean(0); sd_p = tr[1].std(0) + 1e-6

    out = {"MLP (centre, coast-blind)": [], "GNN (7x7, sees coast)": []}
    for s in seeds:
        out["MLP (centre, coast-blind)"].append(train_one(MLPBeach, tr, te, mu_o, sd_o, mu_p, sd_p, dev, s))
        out["GNN (7x7, sees coast)"].append(train_one(GNNBeach, tr, te, mu_o, sd_o, mu_p, sd_p, dev, s))
        print(f"  seed {s}: MLP AUC={out['MLP (centre, coast-blind)'][-1]['overall'][0]:.3f}  "
              f"GNN AUC={out['GNN (7x7, sees coast)'][-1]['overall'][0]:.3f}")

    print("\n=== BEACHING RESULT — held-out particles (ROC-AUC / avg-precision, mean±std) ===")
    for model in out:
        for cls in ("overall", "micro", "macro"):
            aucs = np.array([r[cls][0] for r in out[model] if cls in r])
            aps = np.array([r[cls][1] for r in out[model] if cls in r])
            print(f"  {model:26s} {cls:8s}  AUC={aucs.mean():.3f}±{aucs.std():.3f}  AP={aps.mean():.3f}")
    mlp = np.array([r["overall"][0] for r in out["MLP (centre, coast-blind)"]]).mean()
    gnn = np.array([r["overall"][0] for r in out["GNN (7x7, sees coast)"]]).mean()
    print(f"\n  MLP AUC {mlp:.3f}  vs  GNN AUC {gnn:.3f}  ->  GNN gain {gnn-mlp:+.3f}")
    np.savez(DATA_PROCESSED / "beach_result.npz",
             mlp_auc=mlp, gnn_auc=gnn,
             scores={k: [r["overall"] for r in v] for k, v in out.items()})
    return out


if __name__ == "__main__":
    main()
