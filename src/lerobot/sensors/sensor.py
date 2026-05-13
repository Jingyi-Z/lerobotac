#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Abstract base class for non-camera streaming sensors."""

import abc

import numpy as np

from .configs import SensorConfig


class Sensor(abc.ABC):
    """Base class for non-camera sensors (Hall, force-torque, IMU, etc.).

    Concrete subclasses own their own driver thread, stream format, and any
    calibration logic. The framework only requires that ``get_latest_data``
    return an ``np.ndarray`` of the declared ``shape`` (or ``None`` if data
    isn't yet available, e.g. during calibration).
    """

    def __init__(self, config: SensorConfig):
        self.config = config

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of one observation, excluding batch dimension."""

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def start_continuous_read(self) -> None: ...

    @abc.abstractmethod
    def stop_continuous_read(self) -> None: ...

    @abc.abstractmethod
    def get_latest_data(self) -> np.ndarray | None:
        """Return the most recent observation, or None if not yet available."""

    def wait_for_calibration(self, timeout_s: float = 10.0) -> bool:
        """Block until the sensor is ready to produce calibrated data.

        Default implementation: no calibration required, returns True immediately.
        Sensors that need a startup calibration phase should override this.
        """
        return True

    def metadata(self) -> dict:
        """Return a JSON-serializable snapshot of sensor config + state.

        Used to embed calibration & configuration into dataset ``info.json`` for
        reproducibility. Subclasses can extend with sensor-specific fields.
        """
        return {"type": self.__class__.__name__, "shape": list(self.shape)}