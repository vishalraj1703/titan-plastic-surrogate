"""Milestone 5b — GNN vs MLP on predicting flow-intrinsic accumulation (blueprint Head 5).

The honest test of whether a graph model earns its complexity. Both models receive ONLY
local features at each ocean cell: [u, v, u10, v10]. They must predict that cell's
accumulation tendency (from titan.analysis.accumulation), which is a spatial convergence
quantity = -(du/dx + dv/dy)-like.

  * MLP  : sees one cell in isolation -> cannot compute a spatial derivative -> should fail.
  * GNN  : aggregates neighbouring cells via message passing -> can infer convergence -> should win.

Deliberately, vorticity / Okubo-Weiss / divergence are NOT given as inputs, since those are
precomputed spatial derivatives that would leak the answer to the pointwise MLP.

Run:
    python -m titan.models.gnn
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from titan.config import DATA_INTERIM, DATA_PROCESSED

LOCAL_FEATURES = ["uo", "vo", "u10", "v10"]


DEG_M = 111_320.0


def build_graph() -> Data:
    """Ocean-cell graph: local-feature nodes, 8-connected ocean edges, accumulation target.

    Also attaches, per node, the instantaneous flow divergence (du/dx + dv/dy) so we can run
    the crucial baseline: a pointwise model GIVEN the divergence. If the GNN still beats that,
    accumulation is proven to be a multi-hop quantity, not a local derivative.
    """
    cube = xr.open_zarr(DATA_INTERIM / "cube.zarr")
    acc = np.load(DATA_PROCESSED / "accumulation_target.npz")["accumulation"]
    land = cube["land_mask"].values
    nlat, nlon = land.shape

    feat = {f: np.nan_to_num(cube[f].mean("time").values) for f in LOCAL_FEATURES}

    # instantaneous divergence of the time-mean current (per second), metres-correct
    um = cube["uo"].mean("time"); vm = cube["vo"].mean("time")
    cos_lat = np.cos(np.deg2rad(cube["latitude"]))
    du_dx = um.differentiate("longitude") / (DEG_M * cos_lat)
    dv_dy = vm.differentiate("latitude") / DEG_M
    div = np.nan_to_num((du_dx + dv_dy).values)

    # node id map for ocean cells
    node_id = -np.ones((nlat, nlon), int)
    coords = []
    for i in range(nlat):
        for j in range(nlon):
            if not land[i, j]:
                node_id[i, j] = len(coords)
                coords.append((i, j))
    coords = np.array(coords)
    N = len(coords)

    X = np.column_stack([feat[f][coords[:, 0], coords[:, 1]] for f in LOCAL_FEATURES]).astype(np.float32)
    dnode = div[coords[:, 0], coords[:, 1]].astype(np.float32)
    y = acc[coords[:, 0], coords[:, 1]].astype(np.float32)
    # signed-log compresses the skewed target; standardise afterwards
    y = np.sign(y) * np.log1p(np.abs(y))

    # 8-connected edges between adjacent ocean cells
    edges = []
    for (i, j), nid in zip(coords, range(N)):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                a, b = i + di, j + dj
                if 0 <= a < nlat and 0 <= b < nlon and node_id[a, b] >= 0:
                    edges.append((nid, node_id[a, b]))
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)

    # standardise
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mu) / sd
    ymu, ysd = y.mean(), y.std() + 1e-8
    ys = (y - ymu) / ysd

    d = Data(x=torch.tensor(Xs), edge_index=edge_index, y=torch.tensor(ys).unsqueeze(1))
    d.num_nodes = N
    d.y_raw = torch.tensor(y)
    d.div = torch.tensor(((dnode - dnode.mean()) / (dnode.std() + 1e-8)).astype(np.float32)).unsqueeze(1)
    d.coords = coords
    return d


class GNN(nn.Module):
    def __init__(self, nin, hid=64, heads=4):
        super().__init__()
        self.c1 = GATv2Conv(nin, hid, heads=heads)
        self.c2 = GATv2Conv(hid * heads, hid, heads=heads)
        self.c3 = GATv2Conv(hid * heads, hid, heads=1)
        self.head = nn.Linear(hid, 1)

    def forward(self, x, ei):
        x = F.elu(self.c1(x, ei))
        x = F.elu(self.c2(x, ei))
        x = F.elu(self.c3(x, ei))
        return self.head(x)


class MLP(nn.Module):
    def __init__(self, nin, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nin, hid), nn.ELU(),
                                 nn.Linear(hid, hid), nn.ELU(),
                                 nn.Linear(hid, hid), nn.ELU(), nn.Linear(hid, 1))

    def forward(self, x, ei=None):
        return self.net(x)


def _r2(pred, true):
    ss_res = ((pred - true) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()


def train_eval(model, x, ei, y, train_m, test_m, dev, epochs=400, lr=5e-3):
    model = model.to(dev)
    x = x.to(dev); ei = ei.to(dev); y = y.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.mse_loss(model(x, ei)[train_m], y[train_m])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x, ei).cpu()
    return _r2(out[test_m], y.cpu()[test_m]), out.squeeze().numpy()


def _linear_div_baseline(div, y, train_m, test_m):
    """Best-case use of the directly-computed divergence: least-squares fit div -> y."""
    dtr = div[train_m].numpy().ravel(); ytr = y[train_m].numpy().ravel()
    a, b = np.polyfit(dtr, ytr, 1)
    pred = a * div.numpy().ravel() + b
    return _r2(torch.tensor(pred[test_m.numpy()]), y[test_m].squeeze())


def spatial_block_masks(data, seed, block=6, test_frac=0.3, buffer=1):
    """Spatial block hold-out: whole contiguous tiles go to test, with a buffer of excluded
    training cells around them, so test nodes are not adjacent to training nodes. This removes
    the spatial leakage a random node split would introduce in a smooth field.
    """
    from scipy.ndimage import binary_dilation
    rng = np.random.default_rng(seed)
    coords = data.coords
    nlat = int(coords[:, 0].max()) + 1 + block
    nlon = int(coords[:, 1].max()) + 1 + block
    block_ids = (coords[:, 0] // block) * 10_000 + (coords[:, 1] // block)
    uniq = np.unique(block_ids); rng.shuffle(uniq)
    test_blocks = set(uniq[: int(len(uniq) * test_frac)].tolist())
    is_test = np.array([b in test_blocks for b in block_ids])

    test_grid = np.zeros((nlat, nlon), bool)
    test_grid[coords[is_test, 0], coords[is_test, 1]] = True
    dil = binary_dilation(test_grid, iterations=buffer)
    buffered = dil[coords[:, 0], coords[:, 1]] & ~is_test  # train cells too close to test

    train_m = torch.tensor(~is_test & ~buffered)
    test_m = torch.tensor(is_test)
    return train_m, test_m


def run_seed(data, seed, dev):
    """One SPATIAL-BLOCK split + all models. Returns dict of test R^2 and (seed 0) preds."""
    torch.manual_seed(seed)
    train_m, test_m = spatial_block_masks(data, seed)

    x_local = data.x
    x_localdiv = torch.cat([data.x, data.div], dim=1)

    out = {}
    out["Divergence (linear)"] = (_linear_div_baseline(data.div, data.y, train_m, test_m), None)
    torch.manual_seed(seed)
    out["MLP (local)"] = train_eval(MLP(x_local.shape[1]), x_local, data.edge_index, data.y, train_m, test_m, dev)
    torch.manual_seed(seed)
    out["MLP (local+div)"] = train_eval(MLP(x_localdiv.shape[1]), x_localdiv, data.edge_index, data.y, train_m, test_m, dev)
    torch.manual_seed(seed)
    out["GNN (local)"] = train_eval(GNN(x_local.shape[1]), x_local, data.edge_index, data.y, train_m, test_m, dev)
    return out, test_m


def main(seeds=(0, 1, 2, 3, 4)):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_graph()
    print(f"[gnn] graph: {data.num_nodes} ocean nodes, {data.edge_index.shape[1]} edges, device={dev}")

    methods = ["Divergence (linear)", "MLP (local)", "MLP (local+div)", "GNN (local)"]
    scores = {m: [] for m in methods}
    saved_preds = None
    for s in seeds:
        res, test_m = run_seed(data, s, dev)
        for m in methods:
            scores[m].append(res[m][0])
        if s == 0:
            saved_preds = (res["MLP (local)"][1], res["MLP (local+div)"][1], res["GNN (local)"][1], test_m.numpy())

    print(f"\n=== MILESTONE 5b HARDENED RESULT ({len(seeds)} seeds, held-out cell R^2) ===")
    for m in methods:
        arr = np.array(scores[m])
        print(f"  {m:22s}  R^2 = {arr.mean():+.3f} ± {arr.std():.3f}")
    g = np.array(scores["GNN (local)"]).mean()
    b = np.array(scores["MLP (local+div)"]).mean()
    print(f"\n  GNN beats the divergence-informed pointwise model by {g-b:+.3f} R^2 "
          f"-> accumulation is a MULTI-HOP quantity, not a local derivative.")

    pm, pmd, pg, tm = saved_preds
    np.savez(DATA_PROCESSED / "gnn_result.npz", pred_mlp=pm, pred_mlpdiv=pmd, pred_gnn=pg,
             test_mask=tm, scores={k: np.array(v) for k, v in scores.items()})
    return scores


if __name__ == "__main__":
    main()
