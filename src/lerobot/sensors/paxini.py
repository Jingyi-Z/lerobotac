#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Paxini PX-6AX GEN3 tactile sensor adapter for the lerobot sensor framework.

Wraps :class:`paxini_sdk.HighSpeedHandBoard` so a Paxini fingertip / finger-pad
can be used wherever a :class:`lerobot.sensors.Sensor` is expected.

Two output modes (selected via the config's ``output_format`` field):

* ``"resultant"`` — ``(N, 3)`` float32, the resultant Fx, Fy, Fz on the
  active module. Newtons.
* ``"distributed"`` — ``(N, P, 3)`` float32, the per-taxel Fx, Fy, Fz grid.
  P is the active sensor's point count (52 for DP-S2015-Elite).
* ``"both"`` — flattens to ``(N, 3 + P*3)`` so a single sensor entry covers
  the entire force tensor for downstream collection.

The Paxini sensor does its own baseline subtraction in firmware (the
``calibrate()`` call documented in the protocol PDF §1.3.1 and Hand_UI.py
line 274); ``auto_calibrate=True`` runs that at connect time. An optional
host-side baseline can be layered on top via ``software_baseline_frames``.
"""

import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np

from .configs import PaxiniSensorConfig
from .sensor import Sensor


class PaxiniSensor(Sensor):
    """Driver for a Paxini PX-6AX GEN3 tactile sensor via paxini-sdk."""

    def __init__(self, config: PaxiniSensorConfig):
        super().__init__(config)
        self.config: PaxiniSensorConfig = config

        # Lazy import so paxini-sdk is only required if this class is instantiated.
        try:
            from paxini_sdk import HighSpeedHandBoard  # noqa: F401
            from paxini_sdk import registers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "paxini-sdk is not installed. Install it with: pip install paxini-sdk"
            ) from e

        self._hand = None
        self._is_connected = False
        self._stream_iter = None
        self._stop_event = threading.Event()
        self._data_thread: threading.Thread | None = None
        self._data_lock = threading.Lock()
        self._ring: Deque[np.ndarray] = deque(maxlen=config.buffer_size)
        self._sample_dim: int | None = None      # set once we know the shape
        self._active_module_idx: int | None = None
        self._n_taxels: int | None = None

        # Optional software baseline (atop the firmware calibration)
        self._sw_baseline: Optional[np.ndarray] = None
        self._sw_baseline_buf: list[np.ndarray] = []

    # ---- properties --------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def shape(self) -> tuple[int, ...]:
        if self._sample_dim is None:
            # Best-effort: assume resultant if we haven't connected yet.
            # The actual shape is finalized in connect().
            return (self.config.buffer_size, 3)
        if isinstance(self._sample_dim, tuple):
            # Distributed mode: _sample_dim is (P, 3)
            return (self.config.buffer_size, *self._sample_dim)
        return (self.config.buffer_size, self._sample_dim)

    @property
    def is_calibrated(self) -> bool:
        if self.config.software_baseline_frames <= 0:
            return self._is_connected
        return self._sw_baseline is not None

    # ---- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        from paxini_sdk import HighSpeedHandBoard, registers

        self._hand = HighSpeedHandBoard(
            self.config.port, baudrate=self.config.baud_rate
        )
        self._hand.open()
        active = self._hand.read_active_modules()
        if not active:
            self._hand.close()
            self._hand = None
            raise RuntimeError(
                "No active Paxini modules detected. Check the FPC cable orientation."
            )

        if self.config.module_index is None:
            self._active_module_idx = active[0]
        else:
            if self.config.module_index not in active:
                self._hand.close()
                self._hand = None
                raise RuntimeError(
                    f"Requested module {self.config.module_index} "
                    f"({registers.MODULE_NAMES[self.config.module_index]}) "
                    f"not active; active list is {active}."
                )
            self._active_module_idx = self.config.module_index

        # Discover sample dimensionality from the active module's point count
        self._n_taxels = self._hand.read_distribution_point_count(self._active_module_idx)
        if self.config.output_format == "resultant":
            self._sample_dim = 3
        elif self.config.output_format == "distributed":
            self._sample_dim = (self._n_taxels, 3)  # type: ignore[assignment]
        elif self.config.output_format == "both":
            self._sample_dim = 3 + self._n_taxels * 3
        else:
            raise ValueError(
                f"Unsupported output_format: {self.config.output_format!r}"
            )

        if self.config.auto_calibrate:
            logging.info("Paxini: triggering firmware calibration ...")
            self._hand.calibrate()

        self._is_connected = True
        module_name = registers.MODULE_NAMES[self._active_module_idx]
        logging.info(
            f"Connected to Paxini sensor on {self.config.port} "
            f"(module={module_name}, taxels={self._n_taxels}, "
            f"output={self.config.output_format})"
        )

    def disconnect(self) -> None:
        self.stop_continuous_read()
        if self._hand is not None:
            try:
                self._hand.close()
            except Exception:
                pass
        self._hand = None
        self._is_connected = False
        logging.info("Disconnected from Paxini sensor")

    # ---- threading ---------------------------------------------------------

    def start_continuous_read(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Sensor must be connected before start_continuous_read()")
        if self._data_thread and self._data_thread.is_alive():
            return
        self._stop_event.clear()
        self._data_thread = threading.Thread(
            target=self._continuous_read_loop, daemon=True
        )
        self._data_thread.start()
        logging.info("Started Paxini sensor continuous read")

    def stop_continuous_read(self) -> None:
        if self._data_thread and self._data_thread.is_alive():
            self._stop_event.set()
            self._data_thread.join(timeout=2.0)
        self._data_thread = None

    def _continuous_read_loop(self) -> None:
        from paxini_sdk import registers
        module_name = registers.MODULE_NAMES[self._active_module_idx]
        try:
            for frame in self._hand.stream_auto_push(per_frame_timeout=0.2):
                if self._stop_event.is_set():
                    break
                sample = self._frame_to_sample(frame, module_name)
                if sample is None:
                    continue
                # Optional software baseline calibration
                if self.config.software_baseline_frames > 0 and self._sw_baseline is None:
                    self._sw_baseline_buf.append(sample.copy())
                    if len(self._sw_baseline_buf) >= self.config.software_baseline_frames:
                        self._sw_baseline = np.mean(
                            np.stack(self._sw_baseline_buf), axis=0
                        ).astype(np.float32)
                        self._sw_baseline_buf.clear()
                        logging.info(
                            "Paxini: software baseline captured "
                            f"({self.config.software_baseline_frames} frames)"
                        )
                    continue
                if self._sw_baseline is not None:
                    sample = sample - self._sw_baseline
                with self._data_lock:
                    self._ring.append(sample)
        except Exception as e:
            logging.exception(f"Paxini read loop terminated: {e}")

    # ---- data extraction ---------------------------------------------------

    def _frame_to_sample(self, frame, module_name: str) -> np.ndarray | None:
        """Convert one AutoPushFrame into a flat float32 sample of shape
        (sample_dim,) or (P, 3) depending on output_format."""
        if self.config.output_format == "resultant":
            forces = frame.resultant_forces_newtons.get(module_name)
            if forces is None:
                return None
            return np.array(forces, dtype=np.float32)
        elif self.config.output_format == "distributed":
            pts = frame.distributed_forces_newtons.get(module_name, [])
            if not pts:
                return None
            arr = np.array(pts, dtype=np.float32)
            if arr.shape[0] < self._n_taxels:
                pad = np.zeros((self._n_taxels - arr.shape[0], 3), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            return arr  # shape (P, 3)
        elif self.config.output_format == "both":
            res = frame.resultant_forces_newtons.get(module_name)
            pts = frame.distributed_forces_newtons.get(module_name, [])
            if res is None and not pts:
                return None
            res_arr = np.array(res or (0.0, 0.0, 0.0), dtype=np.float32)
            pts_arr = np.array(pts or [(0.0, 0.0, 0.0)] * self._n_taxels,
                                dtype=np.float32)
            if pts_arr.shape[0] < self._n_taxels:
                pad = np.zeros((self._n_taxels - pts_arr.shape[0], 3), dtype=np.float32)
                pts_arr = np.concatenate([pts_arr, pad], axis=0)
            return np.concatenate([res_arr, pts_arr.flatten()], axis=0)
        return None

    def get_latest_data(self) -> np.ndarray | None:
        if not self._is_connected:
            return None
        if not self.is_calibrated:
            return None
        with self._data_lock:
            samples = list(self._ring)
        if not samples:
            return np.zeros(self.shape, dtype=np.float32)

        if self.config.output_format == "distributed":
            # Each sample is (P, 3); stack along axis 0 -> (N, P, 3)
            if len(samples) < self.config.buffer_size:
                pad = np.zeros(
                    (self.config.buffer_size - len(samples), self._n_taxels, 3),
                    dtype=np.float32,
                )
                return np.concatenate([pad, np.stack(samples)], axis=0)
            return np.stack(samples)

        # Resultant / both: each sample is (D,)
        sample_dim = self._sample_dim
        if len(samples) < self.config.buffer_size:
            pad = np.zeros(
                (self.config.buffer_size - len(samples), sample_dim),
                dtype=np.float32,
            )
            return np.concatenate([pad, np.stack(samples)], axis=0)
        return np.stack(samples)

    def wait_for_calibration(self, timeout_s: float = 10.0, poll_s: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self.is_calibrated:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)
        return True

    def metadata(self) -> dict:
        from paxini_sdk import registers
        return {
            **super().metadata(),
            "port": self.config.port,
            "baud_rate": self.config.baud_rate,
            "buffer_size": self.config.buffer_size,
            "module_index": self._active_module_idx,
            "module_name": (
                registers.MODULE_NAMES[self._active_module_idx]
                if self._active_module_idx is not None else None
            ),
            "n_taxels": self._n_taxels,
            "output_format": self.config.output_format,
            "auto_calibrate": self.config.auto_calibrate,
            "software_baseline_frames": self.config.software_baseline_frames,
            "is_calibrated": self.is_calibrated,
            "sw_baseline": (
                self._sw_baseline.tolist() if self._sw_baseline is not None else None
            ),
        }
