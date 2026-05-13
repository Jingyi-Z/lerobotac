#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration for SOSensorFollower."""

from dataclasses import dataclass, field

from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
from lerobot.sensors import SensorConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so_sensor_follower")
@dataclass
class SOSensorFollowerConfig(RobotConfig, SOFollowerConfig):
    """SO Follower (SO-100/101) with arbitrary non-camera sensors.

    Example::

        SOSensorFollowerConfig(
            port="/dev/cu.usbmodemSN234567892",
            sensors={
                "gripper": MLX90393SensorConfig(port="/dev/cu.usbmodem197004501"),
            },
        )
    """

    # Non-camera sensors keyed by name. Each sensor's config carries its own
    # ``type`` (e.g. "mlx90393") which the factory dispatches on.
    sensors: dict[str, SensorConfig] = field(default_factory=dict)