import torch
import torch.nn as nn
from torchdiffeq import odeint

class UTIntegrator1D(nn.Module):
    """Bistable: integrate drift with odeint, add single diffusion kick, UT recombine."""
    def __init__(self, drift_nn: nn.Module, dt=0.01, method="dopri5", rtol=1e-5, atol=1e-7, sigma_val=0.1, include_diffusion=True):
        super().__init__()
        self.f = drift_nn
        self.dt = float(dt); self.method, self.rtol, self.atol = method, rtol, atol
        self.include_diffusion = include_diffusion
        self.sigma_val = sigma_val

    def forward(self, sigma_points: torch.Tensor, W_points: torch.Tensor):
        B, S, D = sigma_points.shape
        device, dtype = sigma_points.device, sigma_points.dtype
        x = sigma_points[..., 0]; u = sigma_points[..., 1]
        w = sigma_points[..., 2] if self.include_diffusion else None

        x0 = x.reshape(-1); u_flat = u.reshape(-1)
        def rhs(t, x_flat):
            inp = torch.stack([x_flat, u_flat], dim=-1)
            return self.f(inp).squeeze(-1)

        tspan = torch.tensor([0.0, self.dt], device=device, dtype=dtype)
        x_end = odeint(rhs, x0, tspan, method=self.method, rtol=self.rtol, atol=self.atol)[-1].view(B, S)

        if self.include_diffusion:
            g2 = 0.5 * (self.sigma_val**2)
            x_end = x_end + torch.sqrt(torch.tensor(2.0*g2*self.dt, device=device, dtype=dtype)) * w

        W = W_points.to(device=device, dtype=dtype)
        return (W * x_end).sum(dim=1, keepdim=True)

class StateIntegratorND(nn.Module):
    """Deterministic integrator for nx>=1 without UT; also used inside UT evolutions."""
    def __init__(self, drift_nn: nn.Module, nx=2, nu=1, dt=0.01, method="dopri5", rtol=1e-5, atol=1e-7):
        super().__init__()
        self.f = drift_nn; self.nx, self.nu = nx, nu
        self.dt = float(dt); self.method, self.rtol, self.atol = method, rtol, atol

    def ode_step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """x:[B,nx] or [nx], u:[B,nu] or [nu] → next x with same leading shape."""
        device, dtype = x.device, x.dtype
        if x.dim() == 1: x = x.unsqueeze(0)
        if u.dim() == 1: u = u.unsqueeze(0)

        def rhs(t, x_flat):
            return self.f(torch.cat([x_flat, u], dim=-1))
        tspan = torch.tensor([0.0, self.dt], device=device, dtype=dtype)
        x_end = odeint(rhs, x, tspan, method=self.method, rtol=self.rtol, atol=self.atol)[-1]
        return x_end if x_end.shape[0] > 1 else x_end.squeeze(0)

    def forward(self, sigma_points: torch.Tensor, W_points: torch.Tensor):
        """UT mean for deterministic nx≥1 (e.g., VDP)."""
        B, S, D = sigma_points.shape
        device, dtype = sigma_points.device, sigma_points.dtype
        x = sigma_points[...,:self.nx].reshape(-1, self.nx)
        u = sigma_points[...,self.nx:].reshape(-1, self.nu)
        def rhs(t, x_flat):
            return self.f(torch.cat([x_flat, u], dim=-1))
        tspan = torch.tensor([0.0, self.dt], device=device, dtype=dtype)
        x_end = odeint(rhs, x, tspan, method=self.method, rtol=self.rtol, atol=self.atol)[-1].view(B,S,self.nx)
        W = W_points.to(device=device, dtype=dtype).unsqueeze(-1)
        return (W * x_end).sum(dim=1)
