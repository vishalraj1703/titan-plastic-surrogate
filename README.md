# TITAN Plastic Surrogate

Physics-trained neural surrogate for OceanParcels marine-plastic transport (Project TITAN, Paper 1) — residual-velocity learning, an analytical windage baseline, cross-season generalization testing, and real-world GPS drifter validation.

Companion code for: *"Emulating a Lagrangian Marine-Plastic Transport Model with a Residual-Velocity Neural Surrogate: Physical Baselines, Real-World Validation, and Cross-Season Generalization"* (submitted to the Journal of Marine Science and Engineering).

## Contents

- `src/titan/` — full pipeline: data preprocessing, OceanParcels physics simulation, dataset construction, surrogate model + training, autoregressive rollout evaluation, and analysis modules (accumulation zones, probabilistic ensemble forecast, GPS-drifter validation).
- `config/region.yaml` — study domain, release sources, particle-class physics parameters, and Copernicus product identifiers.
- `revision_analysis/` — scripts for the analytical windage baseline, bootstrap confidence intervals, the 3-way rollout comparison, the multi-scale computational benchmark, the cross-season generalization test, and quantitative density-field validation.
- `surrogate_mlp.pt` — trained model checkpoint (weights, feature list, standardization statistics) used for every result in the manuscript.

## Reproducing the results

Raw Copernicus current/wind NetCDF files are not redistributed here (freely available from https://marine.copernicus.eu; product identifiers in `config/region.yaml`). Given those files:

1. `python -m titan.data.preprocess` — build the interim data cube
2. `python -m titan.physics.simulate` — generate training + independent-test trajectories
3. `python -m titan.models.dataset` — build residual-velocity training pairs
4. `python -m titan.models.surrogate` — train the surrogate (or use the provided `surrogate_mlp.pt` checkpoint directly)
5. `python -m titan.models.rollout` — reproduce the baseline 72h rollout
6. Run the `revision_analysis/*.py` scripts for the baseline/generalization/benchmark/density analyses.
