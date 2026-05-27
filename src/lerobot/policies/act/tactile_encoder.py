"""Tactile encoders: each maps a dict of tactile tensors to one (B, dim_model) token."""
import math

import torch
from torch import Tensor, nn

from lerobot.configs.types import PolicyFeature


def _safe(key: str) -> str:
    """Make a feature key usable as an nn.ModuleDict key (no dots)."""
    return key.replace(".", "__")


def _flatten(x: Tensor, n_sensor_dims: int) -> Tensor:
    """Flatten the trailing n_sensor_dims of x, keeping leading (batch) dims."""
    leading = x.shape[: x.dim() - n_sensor_dims]
    return x.reshape(*leading, -1)


class _BaseTactileEncoder(nn.Module):
    def __init__(self, sensor_features: dict[str, PolicyFeature], dim_model: int):
        super().__init__()
        self.sensor_keys: list[str] = list(sensor_features.keys())
        self.sensor_shapes: dict[str, tuple[int, ...]] = {
            k: tuple(ft.shape) for k, ft in sensor_features.items()
        }
        self.dim_model = dim_model


class LinearTactileEncoder(_BaseTactileEncoder):
    def __init__(self, sensor_features, dim_model):
        super().__init__(sensor_features, dim_model)
        total = sum(math.prod(s) for s in self.sensor_shapes.values())
        self.proj = nn.Linear(total, dim_model)

    def forward(self, batch):
        flats = [_flatten(batch[k], len(self.sensor_shapes[k])) for k in self.sensor_keys]
        return self.proj(torch.cat(flats, dim=-1))


class MLPTactileEncoder(_BaseTactileEncoder):
    def __init__(self, sensor_features, dim_model, hidden_dim, num_layers, dropout):
        super().__init__(sensor_features, dim_model)
        total = sum(math.prod(s) for s in self.sensor_shapes.values())
        layers, in_dim = [], total
        for _ in range(max(num_layers, 0)):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, dim_model)]
        self.net = nn.Sequential(*layers)

    def forward(self, batch):
        flats = [_flatten(batch[k], len(self.sensor_shapes[k])) for k in self.sensor_keys]
        return self.net(torch.cat(flats, dim=-1))


def _require_2d(sensor_shapes, encoder):
    bad = {k: s for k, s in sensor_shapes.items() if len(s) != 2}
    if bad:
        raise ValueError(
            f"The '{encoder}' tactile encoder requires 2-D tactile features "
            f"(time, axis). Got non-2-D shapes: {bad}."
        )


class CNNTactileEncoder(_BaseTactileEncoder):
    def __init__(self, sensor_features, dim_model, hidden_dim, num_layers, dropout):
        super().__init__(sensor_features, dim_model)
        _require_2d(self.sensor_shapes, "cnn")
        self.subnets = nn.ModuleDict()
        for key in self.sensor_keys:
            blocks, in_ch = [], 1
            for _ in range(max(num_layers, 1)):
                blocks += [nn.Conv2d(in_ch, hidden_dim, 3, padding=1), nn.ReLU(), nn.Dropout(dropout)]
                in_ch = hidden_dim
            blocks += [nn.AdaptiveAvgPool2d(1)]
            self.subnets[_safe(key)] = nn.Sequential(*blocks)
        self.proj = nn.Linear(hidden_dim * len(self.sensor_keys), dim_model)

    def forward(self, batch):
        embeds = []
        for key in self.sensor_keys:
            x = batch[key].unsqueeze(1)                  # (B, time, axis) -> (B, 1, time, axis)
            x = self.subnets[_safe(key)](x)              # (B, hidden, 1, 1)
            embeds.append(x.flatten(start_dim=1))        # (B, hidden)
        return self.proj(torch.cat(embeds, dim=-1))


class TCNTactileEncoder(_BaseTactileEncoder):
    def __init__(self, sensor_features, dim_model, hidden_dim, num_layers, dropout):
        super().__init__(sensor_features, dim_model)
        _require_2d(self.sensor_shapes, "tcn")
        self.subnets = nn.ModuleDict()
        for key in self.sensor_keys:
            n_axis = self.sensor_shapes[key][1]
            blocks, in_ch = [], n_axis
            for layer in range(max(num_layers, 1)):
                d = 2**layer
                blocks += [
                    nn.Conv1d(in_ch, hidden_dim, 3, padding=d, dilation=d),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_ch = hidden_dim
            blocks += [nn.AdaptiveAvgPool1d(1)]
            self.subnets[_safe(key)] = nn.Sequential(*blocks)
        self.proj = nn.Linear(hidden_dim * len(self.sensor_keys), dim_model)

    def forward(self, batch):
        embeds = []
        for key in self.sensor_keys:
            x = batch[key].transpose(1, 2)               # (B, time, axis) -> (B, axis, time)
            x = self.subnets[_safe(key)](x)              # (B, hidden, 1)
            embeds.append(x.flatten(start_dim=1))        # (B, hidden)
        return self.proj(torch.cat(embeds, dim=-1))


class TransformerTactileEncoder(_BaseTactileEncoder):
    def __init__(self, sensor_features, dim_model, hidden_dim, num_layers, dropout):
        super().__init__(sensor_features, dim_model)
        _require_2d(self.sensor_shapes, "transformer")
        n_heads = 4 if hidden_dim % 4 == 0 else 1
        self.input_proj = nn.ModuleDict()
        self.pos_embed = nn.ParameterDict()
        self.encoders = nn.ModuleDict()
        for key in self.sensor_keys:
            seq_len, n_axis = self.sensor_shapes[key]
            self.input_proj[_safe(key)] = nn.Linear(n_axis, hidden_dim)
            self.pos_embed[_safe(key)] = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoders[_safe(key)] = nn.TransformerEncoder(layer, num_layers=max(num_layers, 1))
        self.proj = nn.Linear(hidden_dim * len(self.sensor_keys), dim_model)

    def forward(self, batch):
        embeds = []
        for key in self.sensor_keys:
            x = self.input_proj[_safe(key)](batch[key])  # (B, time, hidden)
            x = x + self.pos_embed[_safe(key)]
            x = self.encoders[_safe(key)](x)
            embeds.append(x.mean(dim=1))
        return self.proj(torch.cat(embeds, dim=-1))


def make_tactile_encoder(encoder, sensor_features, dim_model,
                         hidden_dim=256, num_layers=2, dropout=0.1):
    """Factory: build a tactile encoder by name."""
    if not sensor_features:
        raise ValueError("make_tactile_encoder() requires at least one tactile feature.")
    if encoder == "linear":
        return LinearTactileEncoder(sensor_features, dim_model)
    if encoder == "mlp":
        return MLPTactileEncoder(sensor_features, dim_model, hidden_dim, num_layers, dropout)
    if encoder == "cnn":
        return CNNTactileEncoder(sensor_features, dim_model, hidden_dim, num_layers, dropout)
    if encoder == "tcn":
        return TCNTactileEncoder(sensor_features, dim_model, hidden_dim, num_layers, dropout)
    if encoder == "transformer":
        return TransformerTactileEncoder(sensor_features, dim_model, hidden_dim, num_layers, dropout)
    raise ValueError(f"Unknown tactile encoder {encoder!r}.")