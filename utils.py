import numpy as np
import torch

def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def to_device_dtype(x, device, dtype):
    return torch.as_tensor(x, device=device, dtype=dtype)

def cholesky_psd(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    I = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
    return torch.linalg.cholesky(A + eps * I)
