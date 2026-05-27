"""TactileToEnvStateProcessorStep: flatten tactile -> observation.environment_state."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="tactile_to_env_state_processor")
class TactileToEnvStateProcessorStep(ObservationProcessorStep):
    sensor_keys: list[str] = field(default_factory=list)
    sensor_shapes: list[list[int]] = field(default_factory=list)
    env_state_key: str = OBS_ENV_STATE
    drop_sensor_keys: bool = True

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self.sensor_keys:
            return observation
        obs = dict(observation)
        present = [key for key in self.sensor_keys if key in obs]
        if not present:
            return obs
        if len(present) != len(self.sensor_keys):
            missing = [key for key in self.sensor_keys if key not in obs]
            raise KeyError(
                f"TactileToEnvStateProcessorStep expected all tactile sensor keys "
                f"to be present but {missing} are missing from the observation."
            )
        flats = []
        for key, shape in zip(self.sensor_keys, self.sensor_shapes, strict=True):
            value = obs[key]
            n = len(shape)
            if isinstance(value, torch.Tensor):
                leading = value.shape[: value.dim() - n]
                flat = value.reshape(*leading, -1).to(torch.float32)
            else:
                value = np.asarray(value, dtype=np.float32)
                leading = value.shape[: value.ndim - n]
                flat = torch.from_numpy(value.reshape(*leading, -1))
            flats.append(flat)
        ref = next((f for f in flats if isinstance(f, torch.Tensor)), None)
        flats = [f.to(device=ref.device, dtype=ref.dtype) for f in flats]
        obs[self.env_state_key] = torch.cat(flats, dim=-1)
        if self.drop_sensor_keys:
            for key in present:
                obs.pop(key, None)
        return obs

    def get_config(self) -> dict[str, Any]:
        return {
            "sensor_keys": list(self.sensor_keys),
            "sensor_shapes": [list(s) for s in self.sensor_shapes],
            "env_state_key": self.env_state_key,
            "drop_sensor_keys": self.drop_sensor_keys,
        }

    def transform_features(self, features):
        if not self.sensor_keys:
            return features
        new_features = dict(features)
        obs_features = dict(features.get(PipelineFeatureType.OBSERVATION, {}))
        total_dim = 0
        folded_any = False
        for key, shape in zip(self.sensor_keys, self.sensor_shapes, strict=True):
            if key in obs_features:
                folded_any = True
                obs_features.pop(key)
            total_dim += int(np.prod(shape)) if len(shape) else 1
        if folded_any:
            obs_features[self.env_state_key] = PolicyFeature(type=FeatureType.ENV, shape=(total_dim,))
            new_features[PipelineFeatureType.OBSERVATION] = obs_features
        return new_features