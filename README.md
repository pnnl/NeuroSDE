# StochID — UT-based Neural ODE/SDE System ID (Bistable & Van der Pol)

**note code may not run in all places, and abstractions not fully cleaned**


Lightweight research code for system identification with **Neural ODE drifts** and the **Unscented Transform (UT)**.  
Covers both **stochastic** (bistable SDE) and **deterministic** (Van der Pol ODE) settings.

- Stochastic: learn a drift \(g_{1,\theta}(x,u)\) and evaluate with **whitebox diffusion** \(g_2=\sigma^2/2\).
- Deterministic: learn a drift \(g_{1,\theta}(x,u)\); covariance collapses (UT degenerates to the mean).

---

## Features

- 📦 Datasets for **bistable SDE** and **VDP ODE** with UT-ready sigma clouds
- 🔧 Integrators:
  - `UTIntegrator1D`: drift via `odeint`, diffusion kick, UT recombination (stochastic, 1D)
  - `StateIntegratorND`: drift via `odeint` for \(n_x\ge1\) (deterministic)
- 🧪 Whitebox sanity test that swaps in true dynamics but keeps your UT/integration stack
- 🚀 Rollout utilities for **whitebox** (Euler–Maruyama) and **learned** ensembles
- 📊 Evaluation helpers (MSE/R², RMSE/NRMSE) and (optional) plotting

---

## Repo layout

stochid/
├─ datasets.py # Dataset classes + loaders (bistable/VDP)
├─ eval.py # Whitebox test, compare learned vs whitebox, simple metrics
├─ rollouts.py # Learned and whitebox rollouts (EM / stochastic drift)
├─ Integrators.py # UTIntegrator1D (SDE) and StateIntegratorND (ODE)
├─ simulators.py # Bistable EM simulator; VDP RK4 simulator
├─ ut.py # UT weights and sigma-point builders
├─ utils.py # (expected) cholesky_psd, set_seed, etc.
└─ viz.py # (optional) plot_ensembles, plot_compare_means
└─ others...


> **Note:** `eval.compare_learned_vs_whitebox_bistable` expects `viz.py` with
> `plot_ensembles()` and `plot_compare_means()`. If you don’t want plots, set
> `show_plots=False`.

---

## Requirements

- Python 3.9+
- PyTorch
- `torchdiffeq`
- NumPy
- Matplotlib (for plots)

Install:
```bash
pip install torch torchdiffeq numpy matplotlib


## Quickstart
### 1) Bistable SDE — build dataset + train a drift
```
import torch
from stochid.simulators import simulate_bistable_records
from stochid.datasets import SigmaMeanDatasetBistable, build_loaders
from stochid.Integrators import UTIntegrator1D
import torch.nn as nn

# --- generate records ---
recs = simulate_bistable_records(
    num_trajectories=600, T=40, repeats=512, dt=0.01, lam=0.0, sigma=0.1
)

# --- dataset + loaders ---
ds = SigmaMeanDatasetBistable(recs, include_diffusion=True, device="cpu")
dl_tr, dl_va, dl_te, _ = build_loaders(ds, batch_size=128)

# --- simple drift net (x,u)->dx/dt ---
class DriftMLP1D(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        dims = [2] + [hidden]*layers + [1]
        layers_ = []
        for i in range(len(dims)-2):
            layers_ += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers_ += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers_)
    def forward(self, xu): return self.net(xu).squeeze(-1)

drift = DriftMLP1D()
integrator = UTIntegrator1D(
    drift_nn=drift, dt=0.01, method="dopri5", rtol=1e-5, atol=1e-7,
    sigma_val=0.1, include_diffusion=True
)

# --- train (mean-squared next-mean) ---
opt = torch.optim.Adam(integrator.parameters(), lr=1e-3)
mse = nn.MSELoss()

for epoch in range(10):
    integrator.train(); loss_sum=0; n=0
    for b in dl_tr:
        sp, W, y = b["sigma_points"], b["W_points"], b["mean_f"]
        yhat = integrator(sp, W)
        loss = mse(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()*y.numel(); n += y.numel()
    print(f"epoch {epoch+1:02d} | train MSE {loss_sum/n:.3e}")
```import torch
from stochid.simulators import simulate_bistable_records
from stochid.datasets import SigmaMeanDatasetBistable, build_loaders
from stochid.Integrators import UTIntegrator1D
import torch.nn as nn

# --- generate records ---
recs = simulate_bistable_records(
    num_trajectories=600, T=40, repeats=512, dt=0.01, lam=0.0, sigma=0.1
)

# --- dataset + loaders ---
ds = SigmaMeanDatasetBistable(recs, include_diffusion=True, device="cpu")
dl_tr, dl_va, dl_te, _ = build_loaders(ds, batch_size=128)

# --- simple drift net (x,u)->dx/dt ---
class DriftMLP1D(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        dims = [2] + [hidden]*layers + [1]
        layers_ = []
        for i in range(len(dims)-2):
            layers_ += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers_ += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers_)
    def forward(self, xu): return self.net(xu).squeeze(-1)

drift = DriftMLP1D()
integrator = UTIntegrator1D(
    drift_nn=drift, dt=0.01, method="dopri5", rtol=1e-5, atol=1e-7,
    sigma_val=0.1, include_diffusion=True
)

# --- train (mean-squared next-mean) ---
opt = torch.optim.Adam(integrator.parameters(), lr=1e-3)
mse = nn.MSELoss()

for epoch in range(10):
    integrator.train(); loss_sum=0; n=0
    for b in dl_tr:
        sp, W, y = b["sigma_points"], b["W_points"], b["mean_f"]
        yhat = integrator(sp, W)
        loss = mse(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()*y.numel(); n += y.numel()
    print(f"epoch {epoch+1:02d} | train MSE {loss_sum/n:.3e}")

```
### 2) Whitebox sanity check (bistable)
```
from stochid.eval import whitebox_test_bistable
whitebox_test_bistable(dl_te, dt=0.01, lam=0.0, sigma=0.1, device="cpu")
```

### 3) Learned vs. Whitebox ensemble comparison (bistable)
```
import torch
import torch.nn as nn
from stochid.simulators import simulate_vdp_records
from stochid.datasets import SigmaMeanDatasetVDP, build_loaders
from stochid.Integrators import StateIntegratorND

# --- records ---
recs = simulate_vdp_records(num_trajectories=400, T=200, dt=0.01, mu_vdp=1.0)

# --- dataset + loaders ---
ds = SigmaMeanDatasetVDP(recs, device="cpu")
dl_tr, dl_va, dl_te, _ = build_loaders(ds, batch_size=128)

# --- drift net (x1,x2,u)->[dx1,dx2] ---
class DriftMLP2D(nn.Module):
    def __init__(self, hidden=128, layers=3):
        super().__init__()
        dims=[3]+[hidden]*layers+[2]
        mods=[]
        for i in range(len(dims)-2):
            mods+=[nn.Linear(dims[i],dims[i+1]), nn.ReLU()]
        mods+=[nn.Linear(dims[-2],dims[-1])]
        self.net=nn.Sequential(*mods)
    def forward(self, xu): return self.net(xu)

drift = DriftMLP2D()
integ_vdp = StateIntegratorND(drift_nn=drift, nx=2, nu=1, dt=0.01)

# --- one-step mean training ---
opt = torch.optim.Adam(integ_vdp.parameters(), lr=1e-3)
mse = nn.MSELoss()

for epoch in range(10):
    integ_vdp.train(); loss_sum=0; n=0
    for b in dl_tr:
        sp, W, y = b["sigma_points"], b["W_points"], b["mean_f"]
        yhat = integ_vdp(sp, W)   # [B,2]
        loss = mse(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()*y.numel(); n+=y.numel()
    print(f"epoch {epoch+1:02d} | train MSE {loss_sum/n:.3e}")
```

### 4) Van der Pol (deterministic) — dataset + inference
```
import torch
import torch.nn as nn
from stochid.simulators import simulate_vdp_records
from stochid.datasets import SigmaMeanDatasetVDP, build_loaders
from stochid.Integrators import StateIntegratorND

# --- records ---
recs = simulate_vdp_records(num_trajectories=400, T=200, dt=0.01, mu_vdp=1.0)

# --- dataset + loaders ---
ds = SigmaMeanDatasetVDP(recs, device="cpu")
dl_tr, dl_va, dl_te, _ = build_loaders(ds, batch_size=128)

# --- drift net (x1,x2,u)->[dx1,dx2] ---
class DriftMLP2D(nn.Module):
    def __init__(self, hidden=128, layers=3):
        super().__init__()
        dims=[3]+[hidden]*layers+[2]
        mods=[]
        for i in range(len(dims)-2):
            mods+=[nn.Linear(dims[i],dims[i+1]), nn.ReLU()]
        mods+=[nn.Linear(dims[-2],dims[-1])]
        self.net=nn.Sequential(*mods)
    def forward(self, xu): return self.net(xu)

drift = DriftMLP2D()
integ_vdp = StateIntegratorND(drift_nn=drift, nx=2, nu=1, dt=0.01)

# --- one-step mean training ---
opt = torch.optim.Adam(integ_vdp.parameters(), lr=1e-3)
mse = nn.MSELoss()

for epoch in range(10):
    integ_vdp.train(); loss_sum=0; n=0
    for b in dl_tr:
        sp, W, y = b["sigma_points"], b["W_points"], b["mean_f"]
        yhat = integ_vdp(sp, W)   # [B,2]
        loss = mse(yhat, y)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()*y.numel(); n+=y.numel()
    print(f"epoch {epoch+1:02d} | train MSE {loss_sum/n:.3e}")
```

## Key APIs
```
datasets.py

SigmaMeanDatasetBistable(records, include_diffusion, device, dtype)
Returns batches with:

sigma_points: [B,S,3] = [x, u, w]

W_points: [B,S] (UT mean weights)

mean_f: [B,1] (MC next-step mean)

SigmaMeanDatasetVDP(records, device, dtype)
Returns:

sigma_points: [B,S,3] = [x1,x2,u] (Σ=0 ⇒ repeated mean)

mean_f: [B,2] (next-step deterministic mean)

build_loaders(dataset, batch_size, train_frac, val_frac, shuffle) → (dl_tr, dl_va, dl_te, splits)

Integrators.py

UTIntegrator1D(drift_nn, dt, method, rtol, atol, sigma_val, include_diffusion)
forward(sigma_points, W_points) -> [B,1] (UT mean after drift+kick)

StateIntegratorND(drift_nn, nx, nu, dt, method, rtol, atol)

ode_step(x, u) -> x_next (deterministic step with odeint)

forward(sigma_points, W_points) -> [B,nx] (UT mean, no diffusion)

rollouts.py

rollout_whitebox_bistable_em(x0, u_seq, steps, dt, lam, sigma, particles, seed)
Returns (traj_all: [P,T+1,1], traj_mean: [T+1,1]).

rollout_learned_state_stochastic(integrator, x0, u_seq, steps, sigma_val, particles, seed)
Drift via learned integrator + one Gaussian kick per step.

eval.py

whitebox_test_bistable(dataloader, dt, lam, sigma, device, method, rtol, atol, include_diffusion)
Plugs analytic drift
𝜆
+
𝑥
−
𝑥
3
+
𝑢
λ+x−x
3
+u into the UT pipeline.

compare_learned_vs_whitebox_bistable(...)
Runs both ensembles and (optionally) plots.

ut.py

ut_weights_torch(n, alpha, beta, kappa, device, dtype)

aug_sigma_1d_xw(mu, var, ...) -> (S, Wm) for SDE
[
𝑥
,
𝑤
]
[x,w]

det_sigma_copy(mu_vec, ...) -> (S, Wm) for ODE (Σ=0)

simulators.py

Bistable: simulate_bistable_records(...) → list of (mu_k, var_k, u, mu_{k+1})

VDP: simulate_vdp_records(...) → list of (mu_k, Σ_k=0, u, mu_{k+1})

Helpers: em_step_bistable, vdp_drift_np, rk4_step
```
