from typing import Callable, List, Tuple
import numpy as np
import torch
from .ut import aug_sigma_1d_xw
from .utils import set_seed

# ---------- Bistable SDE (Euler–Maruyama) ----------
def em_step_bistable(x, u, lam, dt, sigma, rng) -> np.ndarray:
    drift = lam + x - x**3 + u
    noise = sigma * rng.normal(size=x.shape)
    return x + drift*dt + noise*np.sqrt(dt)

def simulate_bistable_records(
    num_trajectories=400, T=30, repeats=256, dt=0.01, lam=0.0, sigma=0.1,
    u_sampler: Callable = lambda n: np.random.uniform(-1, 1, size=(n,)),
    x0_sampler: Callable = lambda n: np.random.uniform(-2, 2, size=(n,))
) -> List[Tuple[float, float, float, float]]:
    """Return list of (mu_k, var_k, u, mu_{k+1}) for nx=1."""
    rng = np.random.default_rng()
    recs = []
    for _ in range(num_trajectories):
        u_val = float(u_sampler(1)[0])
        x = x0_sampler(repeats).astype(np.float64)
        X = np.zeros((repeats, T+1)); X[:, 0] = x
        for t in range(T):
            X[:, t+1] = em_step_bistable(X[:, t], u_val, lam, dt, sigma, rng)
        for k in range(T):
            xk, xkp1 = X[:, k], X[:, k+1]
            recs.append((float(xk.mean()), float(xk.var()), u_val, float(xkp1.mean())))
    return recs

# ---------- Van der Pol (deterministic) ----------
def vdp_drift_np(x, u=0.0, mu=1.0):
    x1, x2 = x[..., 0], x[..., 1]
    dx1 = x2
    dx2 = mu*(1.0 - x1**2)*x2 - x1 + u
    return np.stack([dx1, dx2], axis=-1)

def rk4_step(f, x, dt, **kwargs):
    k1 = f(x, **kwargs); k2 = f(x + 0.5*dt*k1, **kwargs)
    k3 = f(x + 0.5*dt*k2, **kwargs); k4 = f(x + dt*k3, **kwargs)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def simulate_vdp_records(
    num_trajectories=400, T=200, dt=0.01, mu_vdp=1.0,
    u_sampler=lambda n: np.random.uniform(-1,1,size=(n,)),
    x0_sampler=lambda n: np.stack([np.random.uniform(-2,2,size=(n,)),
                                   np.random.uniform(-2,2,size=(n,))], axis=-1)
):
    """Deterministic VDP: yields (mu_k, Sigma_k=0, u, mu_{k+1})."""
    recs = []
    for _ in range(num_trajectories):
        u_val = float(u_sampler(1)[0])
        x = x0_sampler(1).astype(np.float64).reshape(2)
        for _ in range(T):
            x_next = rk4_step(vdp_drift_np, x, dt, u=u_val, mu=mu_vdp)
            recs.append((x.astype(np.float32), np.zeros((2,2), np.float32), u_val, x_next.astype(np.float32)))
            x = x_next
    return recs
