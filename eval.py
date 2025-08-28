import torch, numpy as np
from torchdiffeq import odeint

def whitebox_test_bistable(dataloader, dt, lam, sigma, device="cpu",
                           method="dopri5", rtol=1e-5, atol=1e-7, include_diffusion=True):
    def drift(X, U): return lam + X - X**3 + U
    def g2(): return 0.5 * sigma**2

    se, n = 0.0, 0
    for b in dataloader:
        sp, W, y = b["sigma_points"].to(device), b["W_points"].to(device), b["mean_f"].to(device)
        x, u = sp[...,0], sp[...,1]; w = sp[...,2] if include_diffusion else None
        x0 = x.reshape(-1); u_flat = u.reshape(-1)
        def rhs(t, x_flat): return drift(x_flat, u_flat)
        tspan = torch.tensor([0.0, dt], device=device, dtype=sp.dtype)
        x_end = odeint(rhs, x0, tspan, method=method, rtol=rtol, atol=atol)[-1].view_as(x)
        if include_diffusion:
            x_end = x_end + torch.sqrt(torch.tensor(2.0*g2()*dt, device=device, dtype=sp.dtype)) * w
        yhat = (W.to(device=device, dtype=sp.dtype) * x_end).sum(dim=1, keepdim=True)
        se += torch.nn.functional.mse_loss(yhat, y, reduction="sum").item(); n += y.numel()
    mse = se / n
    print(f"[whitebox] MSE: {mse:.4e}")
    return mse

# deprecated 
def rollout_truth_vdp(x0, u_seq, dt, steps, mu=1.0):
    from .simulators import vdp_drift_np, rk4_step
    traj = np.zeros((steps+1, len(x0)), dtype=np.float64); traj[0]=x0
    for k in range(steps):
        traj[k+1] = rk4_step(vdp_drift_np, traj[k], dt, u=float(u_seq[k]), mu=mu)
    return traj.astype(np.float32)

#deprecated 
@torch.no_grad()
def rollout_learned_state(integrator, x0, u_seq, steps=None):
    device = next(integrator.parameters()).device; dtype=next(integrator.parameters()).dtype
    x = torch.as_tensor(x0, device=device, dtype=dtype)
    if x.dim()==1: x = x.unsqueeze(0)
    T = len(u_seq) if steps is None else steps
    traj = torch.zeros(T+1, x.shape[0], x.shape[1], device=device, dtype=dtype)
    traj[0] = x
    for k in range(T):
        u_k = torch.as_tensor(u_seq[k], device=device, dtype=dtype).view(1,-1).repeat(x.shape[0],1)
        x = integrator.ode_step(x, u_k)
        traj[k+1] = x
    return traj.squeeze(1).cpu().numpy()

def pct_error(truth, pred, eps=1e-8):
    truth, pred = np.asarray(truth), np.asarray(pred)
    diff = pred - truth
    rmse = np.sqrt(np.mean(diff**2, axis=0))
    rms  = np.sqrt(np.mean(truth**2, axis=0)) + eps
    return 100.0 * rmse / rms

import numpy as np
import torch
from .rollouts import rollout_whitebox_bistable_em, rollout_learned_state_stochastic
from .metrics import rmse, nrmse_pct
from .viz import plot_ensembles, plot_compare_means

@torch.no_grad()
def compare_learned_vs_whitebox_bistable(
    integrator,
    x0,
    u_seq,
    dt,
    steps=None,
    sigma_val=0.1,     # diffusion used in learned rollout
    lam=0.0,           # whitebox drift tilt
    sigma_white=0.1,   # diffusion in whitebox EM
    particles=64,
    seed=123,
    show_plots=True
):
    """
    Runs:
      - learned ensemble (drift = MLP via odeint, + white-box sigma_val)
      - whitebox ensemble (true drift + EM, + sigma_white)
    and compares mean trajectories.
    """
    T = len(u_seq) if steps is None else steps
    t = np.arange(T+1, dtype=np.float32) * float(dt)

    # Learned ensemble
    learned_all = rollout_learned_state_stochastic(
        integrator, x0, u_seq, steps=T, sigma_val=sigma_val, particles=particles, seed=seed
    )                                       # [P, T+1, 1]
    if learned_all.ndim == 2:               # if particles==1 -> [T+1,1]
        learned_all = learned_all[None, ...]
    learned_mean = learned_all.mean(axis=0) # [T+1, 1]

    # Whitebox ensemble (EM)
    white_all, white_mean = rollout_whitebox_bistable_em(
        x0=x0, u_seq=u_seq, steps=T, dt=dt, lam=lam, sigma=sigma_white,
        particles=particles, seed=seed
    )

    # Metrics on means
    err_rmse   = rmse(learned_mean, white_mean)
    err_nrmse  = nrmse_pct(learned_mean, white_mean)

    if show_plots:
        plot_ensembles(t, traj_white_all=white_all, traj_learn_all=learned_all,
                       mean_white=white_mean, mean_learned=learned_mean,
                       title="Bistable SDE: ensembles & means (whitebox vs learned)")
        plot_compare_means(t, white_mean, learned_mean,
                           title=f"Mean trajectories — RMSE={err_rmse:.3e}, NRMSE={err_nrmse:.2f}%")

    return {
        "rmse_mean": float(err_rmse),
        "nrmse_mean_pct": float(err_nrmse),
        "learned_mean": learned_mean,     # [T+1,1]
        "whitebox_mean": white_mean,      # [T+1,1]
        "learned_all": learned_all,       # [P,T+1,1]
        "white_all": white_all,           # [P,T+1,1]
        "t": t
    }

