#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Factory for instantiating sensors from their configs."""

from .configs import SensorConfig
from .sensor import Sensor


def make_sensors_from_configs(sensor_configs: dict[str, SensorConfig]) -> dict[str, Sensor]:
    """Instantiate a dict of sensors from a dict of sensor configs.

    Dispatches on the registered ``type`` field of each config.
    """
    sensors: dict[str, Sensor] = {}
    for key, cfg in sensor_configs.items():
        if cfg.type == "mlx90393":
            from .mlx90393 import MLX90393Sensor
            sensors[key] = MLX90393Sensor(cfg)
        elif cfg.type == "paxini":
            from .paxini import PaxiniSensor
            sensor = PaxiniSensor(cfg)
            # Tell the sensor the dict key it lives under so its rerun log
            # path can use `observation.sensors.{key}/...`, slotting into the
            # lerobot Blueprint panel auto-built for this sensor.
            sensor.set_sensor_name(key)
            sensors[key] = sensor
        else:
            raise ValueError(
                f"Unknown sensor type {cfg.type!r} for sensor {key!r}. "
                f"Available types: ['mlx90393', 'paxini']"
            )
    return sensors