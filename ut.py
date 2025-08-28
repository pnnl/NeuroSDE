from typing import Tuple
import numpy as np
import torch
from .utils import cholesky_psd

def ut_weights_torch(n: int, alpha=1e-3, beta=2.0, kappa=0.0,
                     device="cpu", dtype=torch.float32) -> Tuple[float, torch.Tensor, torch.Tensor]:
    lam = alpha**2 * (n + kappa) - n
    Wm = torch.zeros(2*n+1, device=device, dtype=dtype)
    Wc = torch.zeros(2*n+1, device=device, dtype=dtype)
    Wm[0] = lam/(n+lam)
    Wc[0] = lam/(n+lam) + (1 - alpha**2 + beta)
    Wm[1:] = Wc[1:] = 1.0/(2*(n+lam))
    return lam, Wm, Wc

def aug_sigma_1d_xw(mu: torch.Tensor, var: torch.Tensor,
                    alpha=1e-3, beta=2.0, kappa=0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """nx=1,nw=1 → sigma points over [x,w]; returns S:[S,2], Wm:[S]."""
    device, dtype = mu.device, mu.dtype
    nx = nw = 1; n = nx + nw
    lam, Wm, _ = ut_weights_torch(n, alpha, beta, kappa, device, dtype)
    Saug = torch.zeros(n, n, device=device, dtype=dtype)
    Saug[0, 0] = var; Saug[1, 1] = 1.0
    L = cholesky_psd((n + lam) * Saug)
    S = torch.zeros(2*n+1, 2, device=device, dtype=dtype)
    S[0, 0] = mu; S[0, 1] = 0.0
    for i in range(n):
        S[1+i,    0] = mu + L[0, i]; S[1+i,    1] =  L[1, i]
        S[1+n+i,  0] = mu - L[0, i]; S[1+n+i,  1] = -L[1, i]
    return S, Wm

def det_sigma_copy(mu_vec: torch.Tensor, alpha=1e-3, beta=2.0, kappa=0.0):
    """Σ=0: copy the mean 2n+1 times and return UT weights."""
    nx = mu_vec.shape[-1]
    _, Wm, _ = ut_weights_torch(nx, alpha, beta, kappa, mu_vec.device, mu_vec.dtype)
    S = mu_vec.reshape(1, nx).repeat(2*nx+1, 1)
    return S, Wm
