from dataclasses import dataclass

@dataclass
class BistableConfig:
    dt: float = 0.01
    lam: float = 0.0
    sigma: float = 0.1
    num_trajectories: int = 600
    T: int = 40
    repeats: int = 512
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cpu"
    ode_method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7

@dataclass
class VDPConfig:
    dt: float = 0.01
    mu: float = 1.0
    num_trajectories: int = 400
    T: int = 200
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    device: str = "cpu"
    ode_method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
