#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""SO Follower robot with arbitrary non-camera sensors."""

import logging
import warnings
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.sensors import make_sensors_from_configs
from lerobot.utils.constants import OBS_SENSORS

from .config_so_sensor_follower import SOSensorFollowerConfig


class SOSensorFollower(SOFollower):
    """SO Follower (SO-100/101) with non-camera sensors keyed by name.

    Each configured sensor produces an observation under the key
    ``observation.sensors.<name>`` with whatever shape the sensor declares.
    """

    config_class = SOSensorFollowerConfig
    name = "so_sensor_follower"

    def __init__(self, config: SOSensorFollowerConfig):
        super().__init__(config)
        self.config: SOSensorFollowerConfig = config

        self._sensors = make_sensors_from_configs(config.sensors)
        for sensor_name, sensor in self._sensors.items():
            try:
                sensor.connect()
                sensor.start_continuous_read()
                logging.info(
                    f"Sensor '{sensor_name}' (type={sensor.config.type}) initialized"
                )
            except Exception as e:
                logging.error(f"Failed to initialize sensor '{sensor_name}': {e}")

    def get_observation(self) -> dict[str, Any]:
        observation = super().get_observation()

        for sensor_name, sensor in self._sensors.items():
            obs_key = f"{OBS_SENSORS}.{sensor_name}"
            try:
                data = sensor.get_latest_data()
                if data is None:
                    warnings.warn(
                        f"Sensor '{sensor_name}' returned no data; substituting zeros.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    data = np.zeros(sensor.shape, dtype=np.float32)
                observation[obs_key] = data
            except Exception as e:
                warnings.warn(
                    f"Sensor '{sensor_name}' read raised {type(e).__name__}: {e}; "
                    "substituting zeros.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                observation[obs_key] = np.zeros(sensor.shape, dtype=np.float32)

        return observation

    @cached_property
    def observation_features(self) -> dict[str, type | tuple | PolicyFeature]:
        features = dict(super().observation_features)
        for sensor_name, sensor in self._sensors.items():
            features[f"{OBS_SENSORS}.{sensor_name}"] = PolicyFeature(
                type=FeatureType.SENSOR,
                shape=sensor.shape,
            )
        return features

    def wait_for_sensor_calibration(self, timeout_s: float = 10.0) -> dict[str, bool]:
        """Block until every sensor reports ready, or timeout per sensor."""
        return {
            name: sensor.wait_for_calibration(timeout_s=timeout_s)
            for name, sensor in self._sensors.items()
        }

    def sensor_metadata(self) -> dict[str, dict]:
        """Per-sensor metadata snapshot, keyed by sensor name."""
        return {name: sensor.metadata() for name, sensor in self._sensors.items()}

    def disconnect(self) -> None:
        for sensor_name, sensor in self._sensors.items():
            try:
                sensor.disconnect()
                logging.info(f"Sensor '{sensor_name}' disconnected")
            except Exception as e:
                logging.error(f"Error disconnecting sensor '{sensor_name}': {e}")
        super().disconnect()