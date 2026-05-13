#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Configuration dataclasses for non-camera sensors (Hall, force-torque, IMU, etc.).

The base ``SensorConfig`` is a draccus ChoiceRegistry; concrete configs register
themselves via ``@SensorConfig.register_subclass("name")`` so the CLI parser
knows how to dispatch them from the ``type: ...`` field.
"""

import abc
from dataclasses import dataclass

import draccus


@dataclass(kw_only=True)
class SensorConfig(draccus.ChoiceRegistry, abc.ABC):  # type: ignore
    """Base configuration class for non-camera streaming sensors."""

    @property
    def type(self) -> str:
        return str(self.get_choice_name(self.__class__))


@SensorConfig.register_subclass("mlx90393")
@dataclass
class MLX90393SensorConfig(SensorConfig):
    """MLX90393 Hall sensor streamed from a Teensy over USB serial."""

    # USB serial port the Teensy appears as on macOS.
    # Find it with `ls /dev/cu.usbmodem*`.
    port: str = "/dev/cu.usbmodem197004501"

    # Must match the baud rate set in the Teensy sketch.
    baud_rate: int = 2_000_000

    # Number of recent samples to keep in the ring buffer (the N in (N, 3)).
    buffer_size: int = 10

    # Number of samples to average for the auto-calibration baseline.
    baseline_frames: int = 100

    # Optional fixed baseline [Bx, By, Bz]. If provided, skips auto-calibration.
    baseline: list[float] | None = None