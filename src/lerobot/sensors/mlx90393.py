#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MLX90393 Hall sensor over Teensy USB serial, with N-sample ring buffer.

Reads 100 Hz binary frames from a Teensy sketch and maintains a rolling
window of the last N samples. Each ``get_latest_data()`` call returns an
``(N, 3)`` float32 array, oldest first.
"""

import logging
import struct
import threading
import time
from collections import deque

import numpy as np
import serial

from .configs import MLX90393SensorConfig
from .sensor import Sensor


class MLX90393Sensor(Sensor):
    """Driver for an MLX90393 Hall sensor connected via a Teensy USB-serial bridge."""

    SYNC0 = 0xAA
    SYNC1 = 0x55
    PAYLOAD_BYTES = 6  # three int16 little-endian values

    def __init__(self, config: MLX90393SensorConfig):
        super().__init__(config)
        self.config: MLX90393SensorConfig = config

        self._baseline: np.ndarray | None = (
            np.asarray(config.baseline, dtype=np.float32)
            if config.baseline is not None
            else None
        )
        self._calib_buffer: list[np.ndarray] = []
        self._ring: deque[np.ndarray] = deque(maxlen=config.buffer_size)

        self._ser: serial.Serial | None = None
        self._is_connected: bool = False
        self._stop_event = threading.Event()
        self._data_thread: threading.Thread | None = None
        self._data_lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.config.buffer_size, 3)

    @property
    def baseline(self) -> np.ndarray | None:
        return self._baseline

    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None

    def connect(self) -> None:
        self._ser = serial.Serial(self.config.port, self.config.baud_rate, timeout=0.1)
        self._is_connected = True
        logging.info(f"Connected to MLX90393 sensor on {self.config.port}")

    def disconnect(self) -> None:
        self.stop_continuous_read()
        if self._ser is not None:
            self._ser.close()
        self._is_connected = False
        logging.info("Disconnected from MLX90393 sensor")

    def _read_one_sample(self) -> np.ndarray | None:
        """Read one frame, resyncing on the 0xAA 0x55 header byte-by-byte."""
        if self._ser is None:
            return None
        b = self._ser.read(1)
        if not b or b[0] != self.SYNC0:
            return None
        b = self._ser.read(1)
        if not b or b[0] != self.SYNC1:
            return None
        payload = self._ser.read(self.PAYLOAD_BYTES)
        if len(payload) != self.PAYLOAD_BYTES:
            return None
        bx, by, bz = struct.unpack("<hhh", payload)
        return np.array([bx, by, bz], dtype=np.float32)

    def _continuous_read_loop(self) -> None:
        while not self._stop_event.is_set():
            raw = self._read_one_sample()
            if raw is None:
                continue
            if self._baseline is None:
                self._calib_buffer.append(raw)
                if len(self._calib_buffer) >= self.config.baseline_frames:
                    self._baseline = np.mean(self._calib_buffer, axis=0).astype(np.float32)
                    self._calib_buffer.clear()
                    logging.info(f"Hall sensor baseline set: {self._baseline}")
                continue
            normalized = raw - self._baseline
            with self._data_lock:
                self._ring.append(normalized)

    def start_continuous_read(self) -> None:
        if self._data_thread and self._data_thread.is_alive():
            return
        self._stop_event.clear()
        self._data_thread = threading.Thread(target=self._continuous_read_loop, daemon=True)
        self._data_thread.start()
        logging.info("Started Hall sensor continuous read")

    def stop_continuous_read(self) -> None:
        if self._data_thread and self._data_thread.is_alive():
            self._stop_event.set()
            self._data_thread.join(timeout=1.0)

    def get_latest_data(self) -> np.ndarray | None:
        if self._baseline is None:
            return None
        with self._data_lock:
            samples = list(self._ring)
        if not samples:
            return np.zeros(self.shape, dtype=np.float32)
        if len(samples) < self.config.buffer_size:
            pad = np.zeros((self.config.buffer_size - len(samples), 3), dtype=np.float32)
            return np.concatenate([pad, np.stack(samples)], axis=0)
        return np.stack(samples)

    def wait_for_calibration(self, timeout_s: float = 10.0, poll_s: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout_s
        while self._baseline is None:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)
        return True

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "port": self.config.port,
            "baud_rate": self.config.baud_rate,
            "buffer_size": self.config.buffer_size,
            "is_calibrated": self._baseline is not None,
            "baseline": self._baseline.tolist() if self._baseline is not None else None,
        }