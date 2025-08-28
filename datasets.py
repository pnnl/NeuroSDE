from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .ut import aug_sigma_1d_xw, det_sigma_copy

class SigmaMeanDatasetBistable(Dataset):
    """Items: sigma_points [S,3]=[x,u,w], W_points [S], mean_f [1]."""
    def __init__(self, records, include_diffusion=True, device="cpu", dtype=torch.float32):
        self.items = []
        for mu_k, var_k, u_val, mu_kp1 in records:
            mu = torch.tensor(mu_k, device=device, dtype=dtype)
            var= torch.tensor(var_k, device=device, dtype=dtype)
            S, Wm = aug_sigma_1d_xw(mu, var)
            x_sigma = S[:, 0:1]; w_sigma = S[:, 1:2]
            u_col = torch.full_like(x_sigma, float(u_val))
            row = torch.cat([x_sigma, u_col, w_sigma], dim=-1) if include_diffusion \
                else torch.cat([x_sigma, u_col], dim=-1)
            self.items.append((row, Wm, torch.tensor([mu_kp1], device=device, dtype=dtype)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        S, Wm, y = self.items[i]
        return {"sigma_points": S, "W_points": Wm, "mean_f": y}

class SigmaMeanDatasetVDP(Dataset):
    """Deterministic; items: sigma_points [S,3]=[x1,x2,u], W [S], mean_f [2]."""
    def __init__(self, records, device="cpu", dtype=torch.float32):
        self.items = []
        for mu_k, _, u_val, mu_kp1 in records:
            mu = torch.tensor(mu_k,    device=device, dtype=dtype)
            S, Wm = det_sigma_copy(mu)
            u_col = torch.full((S.shape[0], 1), float(u_val), device=device, dtype=dtype)
            row   = torch.cat([S, u_col], dim=-1)   # [S,3]
            self.items.append((row, Wm, torch.tensor(mu_kp1, device=device, dtype=dtype)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        S, Wm, y = self.items[i]
        return {"sigma_points": S, "W_points": Wm, "mean_f": y}

def collate_sigma(batch):
    S = torch.stack([b["sigma_points"] for b in batch], 0)
    W = torch.stack([b["W_points"]     for b in batch], 0)
    y = torch.stack([b["mean_f"]       for b in batch], 0)
    return {"sigma_points": S, "W_points": W, "mean_f": y}

def build_loaders(dataset: Dataset, batch_size=128, train_frac=0.7, val_frac=0.15, shuffle=True):
    N = len(dataset)
    idx = np.random.permutation(N)
    n_tr = int(train_frac*N); n_va = int(val_frac*N)
    tr_idx, va_idx, te_idx = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    sub = lambda ds, ids: torch.utils.data.Subset(ds, ids)
    dl_tr = DataLoader(sub(dataset, tr_idx), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_sigma)
    dl_va = DataLoader(sub(dataset, va_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_sigma)
    dl_te = DataLoader(sub(dataset, te_idx), batch_size=batch_size, shuffle=False, collate_fn=collate_sigma)
    return dl_tr, dl_va, dl_te, {"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx}
