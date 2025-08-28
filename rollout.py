import numpy as np
import torch
from torchdiffeq import odeint

# ---------- Whitebox (ground-truth) stochastic rollout via Euler–Maruyama ----------
def rollout_whitebox_bistable_em(x0, u_seq, steps, dt, lam=0.0, sigma=0.1,
                                 particles=64, seed=None):
    """
    dx = (lam + x - x**3 + u) dt + sigma dW
    Returns:
      traj_all  [P, T+1, 1]  (all particle paths)
      traj_mean [T+1, 1]     (ensemble mean)
    """
    if seed is not None:
        np.random.seed(seed)

    x0 = float(np.asarray(x0).reshape(()))
    T  = int(steps)
    u_seq = np.asarray(u_seq, dtype=np.float32)
    assert len(u_seq) == T, "u_seq length must equal steps"

    traj_all = np.zeros((particles, T+1, 1), dtype=np.float32)
    traj_all[:, 0, 0] = x0

    for p in range(particles):
        x = x0
        for k in range(T):
            u = float(u_seq[k])
            drift = lam + x - x**3 + u
            noise = sigma * np.random.randn()
            x = x + drift*dt + noise*np.sqrt(dt)
            traj_all[p, k+1, 0] = x

    traj_mean = traj_all.mean(axis=0)  # [T+1, 1]
    return traj_all, traj_mean

# ---------- Learned stochastic rollout (drift = integrator MLP via odeint, whitebox diffusion) ----------
@torch.no_grad()
def rollout_learned_state_stochastic(integrator, x0, u_seq, steps=None, sigma_val=0.1, particles=1, seed=None):
    """
    Uses integrator.step_state (deterministic drift) and adds a single Gaussian kick per step:
        x_{k+1} = Φ_drift(x_k, u_k; dt) + sqrt(2*g2*dt) * w_k,   g2 = sigma_val^2 / 2
    Args:
      integrator: ODEDriftIntegrator (has .step_state(x,u) and attributes dt/method/rtol/atol)
      x0: scalar or array-like [1]
      u_seq: array-like length T of scalars
      steps: override T
      sigma_val: diffusion magnitude used for the kick
      particles: number of stochastic particles to roll out
      seed: optional RNG seed (numpy)
    Returns: ndarray [T+1, 1] if particles==1, else [P, T+1, 1]
    """
    device = next(integrator.parameters()).device
    dtype  = next(integrator.parameters()).dtype

    if seed is not None:
        np.random.seed(seed)

    u_seq = np.asarray(u_seq, dtype=np.float32)
    T = len(u_seq) if steps is None else steps
    dt = float(integrator.dt)

    def one_particle(x0_scalar, seed_offset=0):
        if seed is not None:
            np.random.seed(seed + int(seed_offset))
        x = torch.tensor([[float(x0_scalar)]], device=device, dtype=dtype)  # [1,1]
        traj = torch.zeros(T+1, 1, device=device, dtype=dtype)
        traj[0, 0] = x.squeeze()

        g2 = 0.5 * (sigma_val**2)
        kick_std = np.sqrt(2.0 * g2 * dt)  # scalar

        for k in range(T):
            u_k = torch.tensor([[float(u_seq[k])]], device=device, dtype=dtype)  # [1,1]
            # drift step via learned ODE
            x = integrator.step_state(x, u_k)  # [1,1]
            # diffusion kick
            w = np.random.randn()
            x = x + torch.tensor([[kick_std * w]], device=device, dtype=dtype)
            traj[k+1, 0] = x.squeeze()

        return traj.detach().cpu().numpy()  # [T+1, 1]

    if particles == 1:
        return one_particle(x0)  # [T+1,1]
    else:
        all_traj = [one_particle(x0, p) for p in range(particles)]
        return np.stack(all_traj, axis=0)  # [P, T+1, 1]
