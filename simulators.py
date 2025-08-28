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



# ---------- OLD CODE FOR CSA ----------
import torch
import torch.nn.functional as F
import numpy as np 

def stoch_dyn_CSA(states):
    """
    Function that simulates stochastic colloidal self-assembly dynamics
      
    states --> [xk, uk, xkw], shape = (3,1) or (3,)
    
    xk --> system state (C6)
    
    uk --> exogenous input (electric field voltage)
    
    xkw --> Gaussian white noise ~ N(0,1)
    
    """
    dt = 1
    
    # Distribute states
    xk = states[0]
    uk = states[1]
    xkw = states[2]
    
    # Get diffusion coefficient
    g2 = 0.0045 * torch.exp(-(xk - 2.1 - 0.75 * uk) ** 2) + 0.0005
    
    # Get drift coefficient
    #    F/KT = 10*(x-2.1-0.75*u)**2
    dFdx = 20 * (xk - 2.1 - 0.75 * uk)
    dg2dx = -2 * (xk - 2.1 - 0.75 * uk) * 0.0045 * torch.exp(-(xk - 2.1 - 0.75 * uk) ** 2)
    g1 = dg2dx - g2 * dFdx
    
    # Predict forward dynamics
    xkp1 = xk + g1 * dt + np.sqrt(2 * g2 * dt) * xkw
    
    return [torch.tensor([xkp1]), 
            torch.tensor([g1]), 
            torch.tensor([g2])]
    # # Sampling time (s)
    # dt = 1
    # # Distribute states
    # xk = states[0]
    # uk = states[1]
    # xkw = states[2]

    
    # # Get diffusion coefficient
    # g2 = 0.0045 * torch.exp(-(xk-2.1-0.75*uk)**2) + 0.0005
    
    # # Get drift coefficient
    # dFdx = 20 * (xk-2.1-0.75*uk)
    # dg2dx = -2 * (xk-2.1-0.75*uk) * 0.0045 * torch.exp(-(xk-2.1-0.75*uk)**2)
    # g1 = dg2dx - g2 * dFdx
    
    # # Predict forward dynamics
    # xkp1 = xk + g1 * dt + torch.sqrt(2 * g2 * dt) * xkw
    
    # return [xkp1.unsqueeze(0), g1.unsqueeze(0), g2.unsqueeze(0)]

def stoch_dyn_LVE(states):
    '''
    Function that simulates stochastic competitive Lotka-Volterra dynamics
    with coexistence equilibrium
    
    states = [xk, yk, xkw, ykw], shape = (4,1) or (4,)
    
    xk, yk --> species populations
    
    xkw, ykw --> independent Gaussian white noise processes, ~ N(0,1)
    
    '''
    # Sampling time (s)
    dt = 0.01
    
    # Distribute states
    xk = states[0]
    yk = states[1]
    xkw = states[2]
    ykw = states[3]
    
    # Enter parameters
    k1 = 0.4
    k2 = 0.5
    xeq = 0.75
    yeq = 0.625
    d1 = 0.5
    d2 = 0.5
    
    # Get drift coefficients
    g1x = xk * (1 - xk - k1 * yk)
    g1y = yk * (1 - yk - k2 * xk)
    
    # Get diffusion coefficients
    g2x = 0.5 * (d1 * xk * (yk - yeq)) ** 2
    g2y = 0.5 * (d2 * yk * (xk - xeq)) ** 2
    
    # Predict forward dynamics
    xkp1 = xk + g1x * dt + torch.sqrt(2 * g2x * dt) * xkw
    ykp1 = yk + g1y * dt + torch.sqrt(2 * g2y * dt) * ykw
    
    return [torch.tensor([[xkp1], [ykp1]]), 
            torch.tensor([[g1x], [g1y]]), 
            torch.tensor([[g2x], [g2y]])]

def stoch_dyn_SIR(states):
    '''
    Function that simulates stochastic Susceptible-Infectious-Recovered (SIR)
    dynamics
    
    states = [sk, ik, rk, skw, ikw, rkw], shape = (6,1) or (6,)
    
    sk, ik, rk --> susceptible, infectious, recovered populations
    
    skw, ikw, rkw --> independent Gaussian white noise processes, ~ N(0,1)
    
    '''
    # Sampling time (s)
    dt = 1
    
    # Distribute states
    sk = states[0]
    ik = states[1]
    rk = states[2]
    skw = states[3]
    ikw = states[4]
    rkw = states[5]
    
    # Enter parameters
    b = 1
    d = 0.1
    k = 0.2
    alpha = 0.5
    gamma = 0.01
    mu = 0.05
    h = 2
    delta = 0.01
    sigma_1 = 0.2
    sigma_2 = 0.2
    sigma_3 = 0.1
    
    # Get nonlinear incidence rate
    g = (k * sk**h * ik) / (sk**h + alpha * ik**h)
    
    # Get drift coefficients
    g1s = b - d * sk - g + gamma * rk
    g1i = g - (d + mu + delta) * ik
    g1r = mu * ik - (d + gamma) * rk
    
    # Get diffusion coefficients
    g2s = 0.5 * (sigma_1 * sk) ** 2
    g2i = 0.5 * (sigma_2 * ik) ** 2
    g2r = 0.5 * (sigma_3 * rk) ** 2
    
    # Predict forward dynamics
    skp1 = sk + g1s * dt + torch.sqrt(2 * g2s * dt) * skw
    ikp1 = ik + g1i * dt + torch.sqrt(2 * g2i * dt) * ikw
    rkp1 = rk + g1r * dt + torch.sqrt(2 * g2r * dt) * rkw
    
    return [torch.tensor([skp1, ikp1, rkp1]), 
            torch.tensor([g1s, g1i, g1r]), 
            torch.tensor([g2s, g2i, g2r]),
            torch.tensor([g]),
            torch.tensor([b - d * sk + gamma * rk]),
            torch.tensor([(d + mu + delta) * ik])]

def stoch_bistable_system(states):
    """Stochastic dynamics for the bistable system."""
    xk = states[0]
    uk = states[1]
    xkw = states[2]
    lam = 0.0
    sigma = 0.5
    dt = 0.01
    g1 = lam + xk - xk**3 + uk
    g2 = (sigma ** 2) / 2
    xkp1 = xk + g1 * dt + torch.sqrt(2 * g2 * dt) * xkw
    return torch.tensor([xkp1]), torch.tensor([g1]), torch.tensor([g2])
