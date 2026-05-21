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
    encoder_type: str = "tcn"
    fz_mode: str = "residual"
    tire_mode: str = "force"
    tire_projection: bool = True
    steering_mode: str = "residual"
    mu_mode: str = "fixed"
    vehicle_mode: str = "residual"
    uncertainty_mode: str = "heteroscedastic"
    vehicle_param_adapter_enabled: bool = False
    residual_bound: float = 0.4
    fz_bound: float = 500.0
    tire_force_bound: float = 2500.0
    tire_moe_expert_count: int = 3
    tire_moe_temperature: float = 1.0
    steering_bound: float = 0.08
    fixed_mu: float = 0.8
    adapter_bound: float = 0.08


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


class GRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(history)
        return h[-1]


class CausalTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, history_len: int) -> None:
        super().__init__()
        nhead = 4 if hidden_dim % 4 == 0 else 2
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, max(1, history_len), hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        length = history.shape[1]
        if length > self.pos.shape[1]:
            pad = self.pos[:, -1:, :].expand(-1, length - self.pos.shape[1], -1)
            pos = torch.cat([self.pos, pad], dim=1)
        else:
            pos = self.pos
        x = self.input_proj(history) + pos[:, :length, :]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=history.device),
            diagonal=1,
        )
        x = self.encoder(x, mask=causal_mask)
        return x[:, -1, :]


def make_encoder(
    encoder_type: str,
    input_dim: int,
    hidden_dim: int,
    history_len: int,
) -> nn.Module:
    normalized = str(encoder_type).lower()
    if normalized in {"e1", "gru"}:
        return GRUEncoder(input_dim, hidden_dim)
    if normalized in {"e2", "tcn", "causal_tcn"}:
        return CausalTCNEncoder(input_dim, hidden_dim)
    if normalized in {"e3", "transformer", "causal_transformer"}:
        return CausalTransformerEncoder(input_dim, hidden_dim, history_len)
    raise ValueError("unknown encoder_type: %s" % encoder_type)


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


def zero_last_linear(module: nn.Module) -> None:
    for layer in reversed(list(module.modules())):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
            return


class HybridStudentModel(nn.Module):
    def __init__(self, config: Optional[HybridStudentConfig] = None) -> None:
        super().__init__()
        self.config = config or HybridStudentConfig()
        c = self.config
        obs_dim = c.state_dim + c.control_dim
        feature_dim = c.hidden_dim + c.state_dim + c.control_dim + c.context_dim
        self.encoder = make_encoder(c.encoder_type, obs_dim, c.hidden_dim, c.history_len)
        self.fz_head = MLPHead(feature_dim, c.hidden_dim, 4)
        self.mu_head = MLPHead(feature_dim, c.hidden_dim, 4)
        self.tire_head = MLPHead(feature_dim + 4, c.hidden_dim, 8)
        self.tire_param_head = MLPHead(feature_dim + 4 + 4, c.hidden_dim, 12)
        tire_moe_expert_count = max(1, int(c.tire_moe_expert_count))
        self.tire_moe_gate = MLPHead(feature_dim + 4 + 4, c.hidden_dim, tire_moe_expert_count)
        self.tire_moe_experts = nn.ModuleList(
            [MLPHead(feature_dim + 4 + 4, c.hidden_dim, 8) for _ in range(tire_moe_expert_count)]
        )
        self.steering_head = MLPHead(feature_dim, c.hidden_dim, 1)
        vehicle_hidden = c.hidden_dim * 2 if c.vehicle_mode in {"large", "V1-large"} else c.hidden_dim
        self.vehicle_head = MLPHead(feature_dim + 13, vehicle_hidden, c.state_dim)
        self.vehicle_param_adapter = MLPHead(c.context_dim, c.hidden_dim, c.state_dim)
        self.uncertainty_head = MLPHead(feature_dim, c.hidden_dim, c.state_dim)
        zero_last_linear(self.vehicle_param_adapter)

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
        fz_nominal = nominal_fz(context)
        if c.fz_mode in {"none", "F0"}:
            fz_delta = torch.zeros_like(fz_nominal)
        else:
            fz_delta = torch.tanh(self.fz_head(features)) * c.fz_bound
        fz = torch.clamp(fz_nominal + fz_delta, min=50.0)
        if c.mu_mode in {"fixed", "M0", "M0-fixed"}:
            mu = fz.new_full(fz.shape, float(c.fixed_mu))
        else:
            mu = 0.05 + 1.45 * torch.sigmoid(self.mu_head(features))
        if c.tire_mode in {"none", "T0"}:
            tire_forces = features.new_zeros(features.shape[0], 8)
            tire_params = features.new_zeros(features.shape[0], 12)
            tire_moe_weights = features.new_zeros(features.shape[0], max(1, int(c.tire_moe_expert_count)))
        elif c.tire_mode in {"param", "T2"}:
            tire_params = torch.tanh(
                self.tire_param_head(torch.cat([features, fz, mu], dim=-1))
            )
            tire_forces = tire_param_forces(
                tire_params,
                current_state,
                current_control,
                fz,
                context,
            )
            tire_moe_weights = features.new_zeros(features.shape[0], max(1, int(c.tire_moe_expert_count)))
        elif c.tire_mode in {"moe", "T3", "T3-moe"}:
            tire_params = features.new_zeros(features.shape[0], 12)
            moe_input = torch.cat([features, fz, mu], dim=-1)
            temperature = max(1.0e-3, float(c.tire_moe_temperature))
            tire_moe_weights = torch.softmax(self.tire_moe_gate(moe_input) / temperature, dim=-1)
            expert_forces = torch.stack(
                [torch.tanh(expert(moe_input)) * c.tire_force_bound for expert in self.tire_moe_experts],
                dim=1,
            )
            tire_forces = (expert_forces * tire_moe_weights[..., None]).sum(dim=1)
        else:
            tire_params = features.new_zeros(features.shape[0], 12)
            tire_raw = torch.tanh(self.tire_head(torch.cat([features, fz], dim=-1)))
            tire_forces = tire_raw * c.tire_force_bound
            tire_moe_weights = features.new_zeros(features.shape[0], max(1, int(c.tire_moe_expert_count)))
        if c.tire_projection:
            tire_forces = project_friction_ellipse(tire_forces, fz, mu)
        if c.steering_mode in {"none", "S0"}:
            steering_delta = features.new_zeros(features.shape[0], 1)
        else:
            steering_delta = torch.tanh(self.steering_head(features)) * c.steering_bound
        vehicle_features = torch.cat(
            [features, fz_delta, tire_forces, steering_delta],
            dim=-1,
        )
        if c.vehicle_mode in {"none", "V0"}:
            delta_x = features.new_zeros(features.shape[0], c.state_dim)
        else:
            delta_x = torch.tanh(self.vehicle_head(vehicle_features)) * c.residual_bound
        if c.vehicle_param_adapter_enabled:
            delta_x = delta_x + torch.tanh(self.vehicle_param_adapter(context)) * c.adapter_bound
        x_next = sanitize_state(physics_next + delta_x)
        if c.uncertainty_mode in {"none", "U0-deterministic"}:
            logvar = torch.zeros_like(x_next)
        else:
            logvar = torch.clamp(self.uncertainty_head(features), min=-8.0, max=5.0)
        return {
            "x_next": x_next,
            "physics_next": physics_next,
            "delta_x": delta_x,
            "fz": fz,
            "fz_delta": fz_delta,
            "mu": mu,
            "tire_forces": tire_forces,
            "tire_params": tire_params,
            "tire_moe_weights": tire_moe_weights,
            "steering_delta": steering_delta,
            "logvar": logvar,
            "z_shared": z,
        }

    def set_trainability(self, fine_tune_mode: str) -> None:
        mode = str(fine_tune_mode or "FT6").upper()
        self.config.vehicle_param_adapter_enabled = mode in {"FT1", "FT6"}
        for param in self.parameters():
            param.requires_grad = False
        if mode == "FT0":
            return
        if mode == "FT6":
            for param in self.parameters():
                param.requires_grad = True
            return
        modules = {
            "FT1": [self.vehicle_param_adapter],
            "FT2": [self.mu_head],
            "FT3": [self.fz_head],
            "FT4": [self.tire_head, self.tire_param_head, self.tire_moe_gate, self.tire_moe_experts],
            "FT5": [self.steering_head],
        }.get(mode)
        if modules is None:
            raise ValueError("unknown fine_tune_mode: %s" % fine_tune_mode)
        for module in modules:
            for param in module.parameters():
                param.requires_grad = True


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
    mu: float | torch.Tensor,
) -> torch.Tensor:
    forces = tire_forces.view(tire_forces.shape[0], 4, 2)
    if torch.is_tensor(mu):
        radius = (fz * mu).clamp_min(1.0)
    else:
        radius = (fz * float(mu)).clamp_min(1.0)
    norm = torch.linalg.norm(forces, dim=-1).clamp_min(1e-6)
    scale = torch.minimum(torch.ones_like(norm), radius / norm)
    return (forces * scale[..., None]).reshape(tire_forces.shape[0], 8)


def tire_param_forces(
    tire_params: torch.Tensor,
    state: torch.Tensor,
    control: torch.Tensor,
    fz: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    radius = context[:, 3:4].clamp_min(1e-6)
    vx = state[:, STATE_KEYS.index("vx") : STATE_KEYS.index("vx") + 1].clamp_min(0.5)
    omega_start = STATE_KEYS.index("omega_fl")
    wheel_omega = state[:, omega_start : omega_start + 4]
    slip_ratio = ((wheel_omega * radius - vx) / vx).clamp(-2.0, 2.0)
    steering_ratio = context[:, 4:5].clamp_min(1e-6)
    steer_proxy = (control[:, 0:1] / steering_ratio).clamp(-0.7, 0.7)
    alpha_front = steer_proxy.expand(-1, 2)
    alpha_rear = state[:, STATE_KEYS.index("vy") : STATE_KEYS.index("vy") + 1] / vx
    slip_angle = torch.cat([alpha_front, alpha_rear.expand(-1, 2)], dim=-1).clamp(-0.7, 0.7)
    d_calpha = tire_params[:, 0:4]
    d_ckappa = tire_params[:, 4:8]
    d_mu = tire_params[:, 8:12]
    c_alpha = 0.18 + 0.12 * d_calpha
    c_kappa = 0.22 + 0.14 * d_ckappa
    mu_scale = (1.0 + 0.4 * d_mu).clamp(0.4, 1.6)
    fx = c_kappa * slip_ratio * fz * mu_scale
    fy = c_alpha * slip_angle * fz * mu_scale
    return torch.stack([fx, fy], dim=-1).reshape(state.shape[0], 8)


def sanitize_state(state: torch.Tensor) -> torch.Tensor:
    columns = []
    omega_keys = {"omega_fl", "omega_fr", "omega_rl", "omega_rr"}
    symmetric_limits = {
        "vy": 80.0,
        "roll": 2.0,
        "pitch": 2.0,
        "yaw": 200.0,
        "p": 20.0,
        "q": 20.0,
        "r": 20.0,
    }
    for idx, key in enumerate(STATE_KEYS):
        value = state[:, idx]
        if key == "vx":
            value = value.clamp_min(0.03)
        elif key in omega_keys:
            value = value.clamp_min(0.0)
        elif key in symmetric_limits:
            value = value.clamp(-symmetric_limits[key], symmetric_limits[key])
        columns.append(value)
    return torch.stack(columns, dim=-1)
