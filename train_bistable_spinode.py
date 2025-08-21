import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint

# ========== utilities ==========
def set_seed(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)

def em_step_bistable(x, u, lam, dt, sigma, rng):
    # dx = (lam + x - x^3 + u) dt + sigma dW
    drift = lam + x - x**3 + u
    noise = sigma * rng.normal(size=x.shape)         
    return x + drift*dt + noise*np.sqrt(dt)

def simulate_records(num_trajectories=400, T=30, repeats=256,
                     dt=0.01, lam=0.0, sigma=0.1,
                     u_sampler=lambda n: np.random.uniform(-1,1,size=(n,)),
                     x0_sampler=lambda n: np.random.uniform(-2,2,size=(n,))):
    """Return list of tuples: (mu_k, var_k, u, mu_{k+1}) for nx=1."""
    rng = np.random.default_rng()
    recs = []
    for _ in range(num_trajectories):
        u_val = float(u_sampler(1)[0])
        x = x0_sampler(repeats).astype(np.float64)
        X = np.zeros((repeats, T+1)); X[:,0] = x
        for t in range(T):
            X[:,t+1] = em_step_bistable(X[:,t], u_val, lam, dt, sigma, rng)
        for k in range(T):
            xk, xkp1 = X[:,k], X[:,k+1]
            recs.append((float(xk.mean()), float(xk.var()), u_val, float(xkp1.mean())))
    return recs

def ut_weights(n, alpha=1e-3, beta=2.0, kappa=0.0, device="cpu", dtype=torch.float32):
    lam = alpha**2*(n+kappa) - n
    Wm = torch.zeros(2*n+1, device=device, dtype=dtype)
    Wc = torch.zeros(2*n+1, device=device, dtype=dtype)
    Wm[0] = lam/(n+lam); Wc[0] = lam/(n+lam) + (1 - alpha**2 + beta)
    Wm[1:] = Wc[1:] = 1.0/(2*(n+lam))
    return lam, Wm, Wc

@torch.no_grad()
def chol_psd(A, eps=1e-12):
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.cholesky(A + eps*I)

def make_aug_sigma(mu, var, alpha=1e-3, beta=2.0, kappa=0.0, device="cpu", dtype=torch.float32):
    """nx=1, nw=1 → sigma points over [x,w]; returns S:[S,2] and Wm:[S]."""
    nx = nw = 1; n = nx + nw
    lam, Wm, _ = ut_weights(n, alpha, beta, kappa, device, dtype)
    Saug = torch.zeros(n, n, device=device, dtype=dtype)
    Saug[0,0] = var; Saug[1,1] = 1.0
    L = chol_psd((n+lam)*Saug)
    S = torch.zeros(2*n+1, 2, device=device, dtype=dtype)
    S[0,0] = mu; S[0,1] = 0.0
    for i in range(n):
        S[1+i,   0] = mu + L[0,i]; S[1+i,   1] =  L[1,i]
        S[1+n+i, 0] = mu - L[0,i]; S[1+n+i, 1] = -L[1,i]
    return S, Wm

def g2_const_sigma(sigma_val, like):
    # 2*g2 = sigma^2 -> g2 = sigma^2/2
    return torch.full_like(like, 0.5*(sigma_val**2))

# ========== dataset ==========
class SigmaMeanDataset(Dataset):
    """
    Each item:
      sigma_points: [S, 3] with columns [x_sigma, u, w_sigma]
      W_points:     [S]
      mean_f:       [1]  (target next-step mean)
    """
    def __init__(self, records, include_diffusion=True, device="cpu", dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.include_diffusion = include_diffusion
        self.items = []
        for mu_k, var_k, u_val, mu_kp1 in records:
            mu = torch.tensor(mu_k, device=device, dtype=dtype)
            var= torch.tensor(var_k, device=device, dtype=dtype)
            S, Wm = make_aug_sigma(mu, var, device=device, dtype=dtype)  # [S,2] over [x,w]
            x_sigma = S[:,0:1]; w_sigma = S[:,1:2]
            u_col = torch.full_like(x_sigma, float(u_val))
            row = torch.cat([x_sigma, u_col, w_sigma], dim=-1) if include_diffusion else torch.cat([x_sigma,u_col], dim=-1)
            self.items.append((row, Wm, torch.tensor([mu_kp1], device=device, dtype=dtype)))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        return {'sigma_points': self.items[i][0], 'W_points': self.items[i][1], 'mean_f': self.items[i][2]}

def collate(batch):
    # pad-free stack (S is constant = 2*(nx+nw)+1 = 5 here)
    sigma_points = torch.stack([b['sigma_points'] for b in batch], 0)  # [B,S,D]
    W_points     = torch.stack([b['W_points']     for b in batch], 0)  # [B,S]
    mean_f       = torch.stack([b['mean_f']       for b in batch], 0)  # [B,1]
    return {'sigma_points': sigma_points, 'W_points': W_points, 'mean_f': mean_f}

# ========== model: Neural ODE drift ==========
class DriftMLP(nn.Module):
    """g1_theta(x,u): [*,2] -> [*,1]"""
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        hs = [hidden]*layers
        dims = [2] + hs + [1]
        mods = []
        for i in range(len(dims)-2):
            mods += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        mods += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*mods)
    def forward(self, xu):
        return self.net(xu)

class ODEDriftIntegrator(nn.Module):
    """
    One-step map for a batch of sigma points:
      - integrate x' = g1_theta(x,u) over dt with odeint (vectorized over B,S)
      - add single diffusion kick sqrt(2 g2 dt) w
      - UT-recombine to next-mean
    """
    def __init__(self, drift_nn: nn.Module, dt=0.01, method='dopri5', rtol=1e-5, atol=1e-7,
                 include_diffusion=True, sigma_val=0.1):
        super().__init__()
        self.f = drift_nn
        self.dt = float(dt)
        self.method, self.rtol, self.atol = method, rtol, atol
        self.include_diffusion = include_diffusion
        self.sigma_val = sigma_val

    def forward(self, sigma_points, W_points):
        """
        sigma_points: [B,S,3] with [x,u,w]; W_points: [B,S]
        returns yhat: [B,1] predicted next mean
        """
        B, S, D = sigma_points.shape
        device, dtype = sigma_points.device, sigma_points.dtype
        x = sigma_points[..., 0]     # [B,S]
        u = sigma_points[..., 1]     # [B,S]
        w = sigma_points[..., 2] if self.include_diffusion else None

        x0 = x.reshape(-1)           # [B*S]
        u_flat = u.reshape(-1)       # constant over [0,dt]

        def rhs(t, x_flat):
            X = x_flat
            inp = torch.stack([X, u_flat], dim=-1)       # [B*S, 2]
            dx = self.f(inp).squeeze(-1)                 # [B*S]
            return dx

        tspan = torch.tensor([0.0, self.dt], device=device, dtype=dtype)
        x_end = odeint(rhs, x0, tspan, method=self.method, rtol=self.rtol, atol=self.atol)[-1]
        x_end = x_end.view(B, S)

        if self.include_diffusion:
            g2 = g2_const_sigma(self.sigma_val, like=x_end)
            x_end = x_end + torch.sqrt(2.0 * g2 * self.dt) * w

        yhat = (W_points.to(device=device, dtype=dtype) * x_end).sum(dim=1, keepdim=True)  # [B,1]
        return yhat

# ========== training loop ==========
def train_neural_ode(
    dt=0.01, lam=0.0, sigma=0.1, seed=0,
    num_trajectories=600, T=40, repeats=512,
    batch_size=128, epochs=20, lr=1e-3, device='cpu'
):
    set_seed(seed)
    # 1) data
    recs = simulate_records(num_trajectories, T, repeats, dt, lam, sigma)
    ds = SigmaMeanDataset(recs, include_diffusion=True, device=device)
    # split
    N = len(ds); idx = np.random.permutation(N)
    n_tr = int(0.7*N); n_va = int(0.15*N)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    sub = lambda ds, ids: torch.utils.data.Subset(ds, ids)
    dl_tr = DataLoader(sub(ds, tr_idx), batch_size=batch_size, shuffle=True, collate_fn=collate)
    dl_va = DataLoader(sub(ds, va_idx), batch_size=batch_size, shuffle=False, collate_fn=collate)
    dl_te = DataLoader(sub(ds, te_idx), batch_size=batch_size, shuffle=False, collate_fn=collate)

    # 2) model
    drift = DriftMLP(hidden=64, layers=2).to(device)
    integrator = ODEDriftIntegrator(drift_nn=drift, dt=dt, method='dopri5', rtol=1e-5, atol=1e-7,
                                    include_diffusion=True, sigma_val=sigma).to(device)
    opt = torch.optim.Adam(integrator.parameters(), lr=lr)
    mse = nn.MSELoss()

    # 3) train
    best = float('inf'); best_state = None
    for epoch in range(1, epochs+1):
        integrator.train(); tr_loss = 0.0
        for batch in dl_tr:
            sp = batch['sigma_points'].to(device)
            W  = batch['W_points'].to(device)
            y  = batch['mean_f'].to(device)          # [B,1]
            yhat = integrator(sp, W)                 # [B,1]
            loss = mse(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()*sp.size(0)
        tr_loss /= len(tr_idx)

        # val
        integrator.eval(); va_loss = 0.0
        with torch.no_grad():
            for batch in dl_va:
                sp = batch['sigma_points'].to(device)
                W  = batch['W_points'].to(device)
                y  = batch['mean_f'].to(device)
                yhat = integrator(sp, W)
                va_loss += mse(yhat, y).item()*sp.size(0)
        va_loss /= len(va_idx)
        print(f"Epoch {epoch:03d} | train {tr_loss:.4e} | val {va_loss:.4e}")
        if va_loss < best:
            best = va_loss
            best_state = {k: v.cpu() for k,v in integrator.state_dict().items()}

    # 4) test
    if best_state is not None:
        integrator.load_state_dict({k: v.to(device) for k,v in best_state.items()})
    integrator.eval(); te_loss = 0.0
    with torch.no_grad():
        for batch in dl_te:
            sp = batch['sigma_points'].to(device)
            W  = batch['W_points'].to(device)
            y  = batch['mean_f'].to(device)
            yhat = integrator(sp, W)
            te_loss += nn.functional.mse_loss(yhat, y, reduction='sum').item()
    te_loss /= len(te_idx)
    print(f"Test MSE: {te_loss:.4e}")
    return integrator, drift, {'train_idx':tr_idx, 'val_idx':va_idx, 'test_idx':te_idx}

# ===== run =====
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    integrator, drift, splits = train_neural_ode(device=device)
