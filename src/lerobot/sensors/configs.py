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

@SensorConfig.register_subclass("paxini")
@dataclass
class PaxiniSensorConfig(SensorConfig):
    """Paxini PX-6AX GEN3 tactile sensor via paxini-sdk.

    Wraps :class:`paxini_sdk.HighSpeedHandBoard` so a Paxini fingertip,
    finger-pad, or palm sensor fits into the lerobot sensor framework. See
    the paxini-sdk docs for full protocol details:
    https://github.com/Jingyi-Z/paxini-sdk
    """

    # Serial port the High-Speed Communication Board appears as.
    # Windows: "COM7". macOS: "/dev/cu.usbserial-XXXX". Linux: "/dev/ttyUSB0".
    port: str = "/dev/cu.usbserial-0001"

    # Default 921600 — the board's fixed UART baud rate.
    baud_rate: int = 921_600

    # Number of recent frames kept in the ring buffer (the N in (N, ...)).
    buffer_size: int = 10

    # Which module slot to read from. None = first active slot reported by
    # the board (correct when only one sensor is plugged in).
    module_index: int | None = None

    # Output tensor shape:
    #   "resultant"   -> (N, 3)         resultant Fx Fy Fz per frame
    #   "distributed" -> (N, P, 3)      per-taxel Fx Fy Fz grid (P from board)
    #   "both"        -> (N, 3 + P*3)   flat concat: resultant + distributed
    output_format: str = "resultant"

    # If True, send the vendor's firmware calibration frame on connect
    # (zeros the Hall-field baseline inside the sensor MCU).
    auto_calibrate: bool = True

    # Optional host-side baseline subtraction stacked on top of firmware
    # calibration. 0 disables. If > 0, average that many frames after
    # connect and subtract from subsequent samples — useful for trimming
    # residual ~1 LSB offset for sub-LSB precision work.
    software_baseline_frames: int = 0

    # If True, the sensor's read thread logs a 3D point-cloud of the 52
    # tactile points (positions from paxini_sdk.sensor_registry) into the
    # active rerun session on every frame. Color = Fz magnitude. Gives the
    # same fingertip-shaped visualization as `paxini-rerun`, inside the
    # `lerobot-record --display_data=true` viewer. Best-effort: if rerun
    # isn't initialized yet, the calls silently no-op.
    display_rerun: bool = False

    # Which communication board the sensor is attached to:
    #   "high_speed" -> High-Speed Communication Board: auto-push stream,
    #                   ~91 Hz, up to 28 modules, reports point counts.
    #   "serial"     -> Serial Converter Board: one sensor, request/response
    #                   only (~8-16 Hz), cannot report its sensor's model.
    board_type: str = "high_speed"

    # Serial Converter Board only: vendor part code of the attached sensor
    # (e.g. "PXSR-STDDP03A"). REQUIRED when board_type="serial" because that
    # board has no point-count register. Ignored for the High-Speed board,
    # which reads the point count from register 0x0030.
    sensor_part_code: str | None = None

    # Serial Converter Board only: slave address = module number + 1.
    # Default 0x01 (module number 0, sensor on the default CN1 port).
    device_id: int = 0x01

    # Lock the sensor's sample rate to a fixed value in Hz -- a stable,
    # deterministic cadence on both boards, the same idea as the Teensy
    # giving the MLX hall sensor a fixed rate.
    #   * Serial Converter Board: the request/response poll loop sleeps the
    #     remainder of each 1/rate period.
    #   * High-Speed board: the ~91 Hz auto-push stream is downsampled --
    #     exactly one frame per 1/rate period is recorded, the rest dropped.
    # 0.0 = use the board's native rate (serial: as fast as it answers;
    # high-speed: the full ~91 Hz auto-push rate).
    poll_rate_hz: float = 0.0
