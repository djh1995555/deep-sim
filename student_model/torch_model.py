from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from student_model.constants import CONTROL_KEYS, STATE_KEYS


@dataclass
class HybridStudentConfig:
    state_dim: int = len(STATE_KEYS)
    control_dim: int = len(CONTROL_KEYS)
    context_dim: int = 17
    hidden_dim: int = 128
    history_len: int = 8
    residual_bound: float = 0.4
    fz_bound: float = 500.0
    tire_force_bound: float = 2500.0
    steering_bound: float = 0.08
    fixed_mu: float = 0.8


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        x = self.norm(x)
        return torch.relu(x + residual)


class CausalTCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            CausalConvBlock(hidden_dim, kernel_size=3, dilation=1),
            CausalConvBlock(hidden_dim, kernel_size=3, dilation=2),
            CausalConvBlock(hidden_dim, kernel_size=3, dilation=4),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = history.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        return x[:, :, -1]


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridStudentModel(nn.Module):
    def __init__(self, config: Optional[HybridStudentConfig] = None) -> None:
        super().__init__()
        self.config = config or HybridStudentConfig()
        c = self.config
        obs_dim = c.state_dim + c.control_dim
        feature_dim = c.hidden_dim + c.state_dim + c.control_dim + c.context_dim
        self.encoder = CausalTCNEncoder(obs_dim, c.hidden_dim)
        self.fz_head = MLPHead(feature_dim, c.hidden_dim, 4)
        self.tire_head = MLPHead(feature_dim + 4, c.hidden_dim, 8)
        self.steering_head = MLPHead(feature_dim, c.hidden_dim, 1)
        self.vehicle_head = MLPHead(feature_dim + 13, c.hidden_dim, c.state_dim)
        self.uncertainty_head = MLPHead(feature_dim, c.hidden_dim, c.state_dim)

    def forward(
        self,
        observable_history: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
        current_control: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        dt: float | torch.Tensor = 0.04,
    ) -> Dict[str, torch.Tensor]:
        c = self.config
        if current_state is None:
            current_state = observable_history[:, -1, : c.state_dim]
        if current_control is None:
            current_control = observable_history[:, -1, c.state_dim :]
        if context is None:
            context = observable_history.new_zeros(
                observable_history.shape[0],
                c.context_dim,
            )
        if not torch.is_tensor(dt):
            dt = observable_history.new_full((observable_history.shape[0],), float(dt))
        if dt.ndim == 0:
            dt = dt.expand(observable_history.shape[0])

        z = self.encoder(observable_history)
        features = torch.cat([z, current_state, current_control, context], dim=-1)
        physics_next = nominal_physics_step(current_state, current_control, context, dt)
        fz_delta = torch.tanh(self.fz_head(features)) * c.fz_bound
        fz_nominal = nominal_fz(context)
        fz = torch.clamp(fz_nominal + fz_delta, min=50.0)
        tire_raw = torch.tanh(self.tire_head(torch.cat([features, fz], dim=-1)))
        tire_forces = tire_raw * c.tire_force_bound
        tire_forces = project_friction_ellipse(tire_forces, fz, c.fixed_mu)
        steering_delta = torch.tanh(self.steering_head(features)) * c.steering_bound
        vehicle_features = torch.cat(
            [features, fz_delta, tire_forces, steering_delta],
            dim=-1,
        )
        delta_x = torch.tanh(self.vehicle_head(vehicle_features)) * c.residual_bound
        x_next = sanitize_state(physics_next + delta_x)
        logvar = torch.clamp(self.uncertainty_head(features), min=-8.0, max=5.0)
        return {
            "x_next": x_next,
            "physics_next": physics_next,
            "delta_x": delta_x,
            "fz": fz,
            "fz_delta": fz_delta,
            "tire_forces": tire_forces,
            "steering_delta": steering_delta,
            "logvar": logvar,
            "z_shared": z,
        }


def nominal_physics_step(
    state: torch.Tensor,
    control: torch.Tensor,
    context: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    wheelbase = context[:, 0].clamp_min(1e-6)
    radius = context[:, 3].clamp_min(1e-6)
    steering_ratio = context[:, 4].clamp_min(1e-6)
    mass = context[:, 5].clamp_min(1.0)
    tau_steer = context[:, 11].clamp_min(0.05)
    dt_col = dt[:, None]
    next_state = state.clone()
    tau_drv = control[:, 4:8].sum(dim=-1)
    tau_brk = control[:, 8:12].sum(dim=-1)
    fx = (tau_drv - tau_brk) / radius - 0.012 * mass * 9.81
    ax = (fx / mass).clamp(-9.0, 5.0)
    sw = control[:, 0]
    delta = (sw / steering_ratio).clamp(-0.6, 0.6)
    vx = state[:, STATE_KEYS.index("vx")].clamp_min(0.05)
    r_target = vx * torch.tan(delta) / wheelbase
    r_prev = state[:, STATE_KEYS.index("r")]
    r_next = r_prev + dt / tau_steer * (r_target - r_prev)
    next_state[:, STATE_KEYS.index("vx")] = (vx + ax * dt).clamp_min(0.05)
    next_state[:, STATE_KEYS.index("vy")] = (
        0.92 * state[:, STATE_KEYS.index("vy")] + 0.08 * vx * torch.sin(delta)
    )
    next_state[:, STATE_KEYS.index("r")] = r_next
    next_state[:, STATE_KEYS.index("yaw")] = state[:, STATE_KEYS.index("yaw")] + r_next * dt
    next_state[:, STATE_KEYS.index("roll")] = 0.96 * state[:, STATE_KEYS.index("roll")]
    next_state[:, STATE_KEYS.index("pitch")] = 0.96 * state[:, STATE_KEYS.index("pitch")]
    next_state[:, STATE_KEYS.index("p")] = (
        next_state[:, STATE_KEYS.index("roll")] - state[:, STATE_KEYS.index("roll")]
    ) / dt.clamp_min(1e-6)
    next_state[:, STATE_KEYS.index("q")] = (
        next_state[:, STATE_KEYS.index("pitch")] - state[:, STATE_KEYS.index("pitch")]
    ) / dt.clamp_min(1e-6)
    wheel_alpha = ((control[:, 4:8] - control[:, 8:12]) / 1.25).clamp(-4000.0, 4000.0)
    omega_start = STATE_KEYS.index("omega_fl")
    next_state[:, omega_start : omega_start + 4] = (
        state[:, omega_start : omega_start + 4] + wheel_alpha * dt_col
    ).clamp_min(0.0)
    return sanitize_state(next_state)


def nominal_fz(context: torch.Tensor) -> torch.Tensor:
    mass = context[:, 5].clamp_min(1.0)
    wheelbase = context[:, 0].clamp_min(1e-6)
    cg_x = context[:, 9]
    front_static = mass * 9.81 * (wheelbase - cg_x) / wheelbase
    rear_static = mass * 9.81 * cg_x / wheelbase
    return torch.stack(
        [
            front_static * 0.5,
            front_static * 0.5,
            rear_static * 0.5,
            rear_static * 0.5,
        ],
        dim=-1,
    )


def project_friction_ellipse(
    tire_forces: torch.Tensor,
    fz: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    forces = tire_forces.view(tire_forces.shape[0], 4, 2)
    radius = (fz * float(mu)).clamp_min(1.0)
    norm = torch.linalg.norm(forces, dim=-1).clamp_min(1e-6)
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return (forces * scale[..., None]).reshape(tire_forces.shape[0], 8)


def sanitize_state(state: torch.Tensor) -> torch.Tensor:
    out = state.clone()
    out[:, STATE_KEYS.index("vx")] = out[:, STATE_KEYS.index("vx")].clamp_min(0.03)
    for key in ["omega_fl", "omega_fr", "omega_rl", "omega_rr"]:
        out[:, STATE_KEYS.index(key)] = out[:, STATE_KEYS.index(key)].clamp_min(0.0)
    for key, limit in [
        ("vy", 80.0),
        ("roll", 2.0),
        ("pitch", 2.0),
        ("yaw", 200.0),
        ("p", 20.0),
        ("q", 20.0),
        ("r", 20.0),
    ]:
        out[:, STATE_KEYS.index(key)] = out[:, STATE_KEYS.index(key)].clamp(
            -limit,
            limit,
        )
    return out
