#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Paxini PX-6AX GEN3 tactile sensor adapter for the lerobot sensor framework.

Wraps the paxini-sdk board drivers so a Paxini fingertip / finger-pad can be
used wherever a :class:`lerobot.sensors.Sensor` is expected.

Two communication boards are supported, selected via the config's
``board_type`` field:

* ``"high_speed"`` -- High-Speed Communication Board. Auto-push streaming at
  ~91 Hz; reports its sensor's point count from a register. Up to 28 modules
  (finger segments + palm) can be plugged into one board.
* ``"serial"``     -- Serial Converter Board. Request/response only (~8-16 Hz,
  no auto-push); cannot report the sensor model, so ``sensor_part_code`` must
  be supplied in the config.

Multiple sensors on one High-Speed board
----------------------------------------
The High-Speed board can carry several sensors at once (e.g. two fingertips
in the "middle finger" and "little finger" slots). To record them together,
declare one ``--robot.sensors`` entry per module, all pointing at the SAME
port but with different ``module_index`` values::

    --robot.sensors='{
      paxini_gripper:    {type: paxini, port: COM10, module_index: 10, ...},
      paxini_wrist_roll: {type: paxini, port: COM10, module_index: 18, ...}
    }'

Under the hood every ``PaxiniSensor`` on the same port shares ONE
``HighSpeedHandBoard`` and ONE auto-push stream (a serial port can only be
opened once, and one stream can only be consumed once). Each sensor pulls its
own module's forces out of every shared frame. See ``_SharedHighSpeedBoard``.

Three output modes (config ``output_format``):

* ``"resultant"``   -- ``(N, 3)`` float32, resultant Fx, Fy, Fz. Newtons.
* ``"distributed"`` -- ``(N, P, 3)`` float32, per-taxel Fx, Fy, Fz grid.
  P is the sensor's point count.
* ``"both"``        -- ``(N, 3 + P*3)`` flat concat of resultant + distributed.

Both boards do firmware baseline subtraction via a ``calibrate()`` call;
``auto_calibrate=True`` runs it at connect time. An optional host-side
baseline can be layered on top via ``software_baseline_frames``.
"""

import json
import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .configs import PaxiniSensorConfig
from .sensor import Sensor

# Newtons per LSB for the Serial Converter Board's raw force counts
# (vendor USB_UI.py parse_resultant_force / parse_distributed_force).
_SERIAL_LSB_N = 0.1

# A resultant reading in newtons, or None if unavailable this cycle.
_ResN = Optional[Tuple[float, float, float]]
# A distributed reading: list of per-taxel (Fx, Fy, Fz) newton tuples.
_DistN = Optional[List[Tuple[float, float, float]]]


# ===========================================================================
# Shared High-Speed board — one board + one stream per physical serial port
# ===========================================================================

_SHARED_REGISTRY: "Dict[Tuple[str, int], _SharedHighSpeedBoard]" = {}
_SHARED_REGISTRY_LOCK = threading.Lock()


class _SharedHighSpeedBoard:
    """One ``HighSpeedHandBoard`` shared by every ``PaxiniSensor`` on the same
    serial port.

    A serial port can only be opened once and its auto-push stream can only be
    consumed by one reader, so when two sensors point at the same port they
    must share a single board and a single stream thread. This class owns:

    * the board object and its lifetime (refcounted; closed when the last
      sensor releases it),
    * one background thread that reads the auto-push stream and fans each
      frame out to every attached per-sensor callback,
    * a one-shot firmware calibration (the board zeroes all active modules at
      once, so it is triggered only once no matter how many sensors ask).

    Instances are obtained via :meth:`acquire` and returned via
    :meth:`release`; never constructed directly by callers.
    """

    def __init__(self, port: str, baudrate: int):
        self.port = port
        self.baudrate = baudrate
        self._board = None
        self._refcount = 0
        self._lock = threading.Lock()          # guards board I/O + flags
        self._readers_lock = threading.Lock()  # guards the callback list
        self._readers: "List[Callable]" = []
        self._stream_stop = threading.Event()
        self._stream_thread: Optional[threading.Thread] = None
        self._calibrated = False
        self._point_counts: Dict[int, int] = {}
        self._claimed: Dict[int, str] = {}     # module_idx -> sensor label
        self.active_modules: List[int] = []

    # ---- acquire / release (refcounted, registry-managed) ------------------

    @classmethod
    def acquire(cls, port: str, baudrate: int) -> "_SharedHighSpeedBoard":
        key = (port, baudrate)
        with _SHARED_REGISTRY_LOCK:
            shared = _SHARED_REGISTRY.get(key)
            if shared is None:
                shared = cls(port, baudrate)
                shared._open()
                _SHARED_REGISTRY[key] = shared
            shared._refcount += 1
            return shared

    def release(self) -> None:
        with _SHARED_REGISTRY_LOCK:
            self._refcount -= 1
            if self._refcount > 0:
                return
            _SHARED_REGISTRY.pop((self.port, self.baudrate), None)
        # Last sensor out: stop the stream and close the board.
        self._stream_stop.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2.0)
        self._stream_thread = None
        if self._board is not None:
            try:
                self._board.close()
            except Exception:
                pass
            self._board = None

    # ---- board setup -------------------------------------------------------

    def _open(self) -> None:
        from paxini_sdk import HighSpeedHandBoard

        self._board = HighSpeedHandBoard(self.port, baudrate=self.baudrate)
        self._board.open()
        self.active_modules = list(self._board.read_active_modules())
        # Pre-read every active module's distributed-force point count NOW,
        # while the board is quiet. Once the auto-push stream starts the port
        # is saturated with AA56 frames, and any later request/response read
        # (e.g. a second sensor calling point_count) collides with the stream
        # -- "short read" / "frame head aa55 not found". Caching all counts up
        # front means every sensor's connect() after the stream is running is
        # a pure cache hit with zero board I/O.
        for m in self.active_modules:
            self._point_counts[m] = self._board.read_distribution_point_count(m)

    def point_count(self, module_idx: int) -> int:
        with self._lock:
            if module_idx in self._point_counts:
                return self._point_counts[module_idx]
            # Not pre-cached (module wasn't in the active list at open). Only
            # safe to read if the auto-push stream hasn't started yet.
            if self._stream_thread is not None and self._stream_thread.is_alive():
                raise RuntimeError(
                    f"Paxini: point count for module {module_idx} was not "
                    "cached before the shared auto-push stream started; cannot "
                    "read it now without corrupting the stream. This usually "
                    "means the module is not in the board's active-module list."
                )
            self._point_counts[module_idx] = (
                self._board.read_distribution_point_count(module_idx)
            )
            return self._point_counts[module_idx]

    def claim_module(self, module_idx: int, label: str) -> None:
        """Record which sensor owns a module; warn on a double-claim (two
        sensors reading the same module on the same board is almost always a
        config mistake — e.g. both left at module_index=None)."""
        with self._lock:
            prev = self._claimed.get(module_idx)
            if prev is not None and prev != label:
                logging.warning(
                    f"Paxini: module {module_idx} is claimed by both "
                    f"{prev!r} and {label!r} on port {self.port}. Both will "
                    "record identical data — set distinct module_index values."
                )
            self._claimed[module_idx] = label

    def request_calibration(self) -> None:
        """Trigger firmware calibration once for the whole board. The board
        zeroes all active modules together, so repeated requests are no-ops.
        Must be called before the stream thread starts (it shares the port)."""
        with self._lock:
            if self._calibrated:
                return
            self._board.calibrate()
            self._calibrated = True

    # ---- stream fan-out ----------------------------------------------------

    def attach_reader(self, callback: Callable) -> None:
        with self._readers_lock:
            if callback not in self._readers:
                self._readers.append(callback)

    def detach_reader(self, callback: Callable) -> None:
        with self._readers_lock:
            if callback in self._readers:
                self._readers.remove(callback)

    def ensure_stream_started(self) -> None:
        with self._lock:
            if self._stream_thread and self._stream_thread.is_alive():
                return
            self._stream_stop.clear()
            self._stream_thread = threading.Thread(
                target=self._stream_loop, daemon=True
            )
            self._stream_thread.start()
            logging.info(
                f"Paxini: shared auto-push stream started on {self.port}"
            )

    def _stream_loop(self) -> None:
        try:
            for frame in self._board.stream_auto_push(per_frame_timeout=0.2):
                if self._stream_stop.is_set():
                    break
                with self._readers_lock:
                    readers = list(self._readers)
                for cb in readers:
                    try:
                        cb(frame)
                    except Exception:
                        logging.exception(
                            "Paxini: shared-board reader callback failed"
                        )
        except Exception as e:
            logging.exception(f"Paxini: shared-board stream terminated: {e}")


class _RawCsvWriter:
    """Per-episode raw-stream sidecar writer, company-schema-compatible.

    One CSV per finger:
      <root>/sensors/<sensor_name>/episode_{i:06d}/<filename {n}>
    Columns exactly match the company's 91 Hz files:
      timestamp_ns, frame_status, time_calibration_offset_ns,
      calibrated_timestamp_ns, fx, fy, fz, p_00_fx, ..., p_{P-1}_fz
    plus an alignment.json with the episode-start epoch time — the anchor
    needed to align the raw stream with the 30 Hz main table exactly
    (the company format only allows first-sample alignment, ~1s off).

    Written from the shared-board stream callback (~91 Hz x n_fingers);
    rows are flushed per write so an aborted episode loses at most one row.
    """

    def __init__(self, root: str, sensor_name: str, episode_index: int,
                 n_fingers: int, n_taxels: int, filename_template: str):
        import os
        self.dir = os.path.join(root, "sensors", sensor_name,
                                f"episode_{episode_index:06d}")
        os.makedirs(self.dir, exist_ok=True)
        header = ["timestamp_ns", "frame_status",
                  "time_calibration_offset_ns", "calibrated_timestamp_ns",
                  "fx", "fy", "fz"]
        for i in range(n_taxels):
            header += [f"p_{i:02d}_fx", f"p_{i:02d}_fy", f"p_{i:02d}_fz"]
        self._files = []
        for k in range(n_fingers):
            path = os.path.join(self.dir, filename_template.format(n=k + 1))
            # utf-8-sig BOM matches the company files byte-for-byte
            f = open(path, "w", encoding="utf-8-sig", newline="")
            f.write(",".join(header) + "\n")
            self._files.append(f)
        with open(os.path.join(self.dir, "alignment.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"episode_start_timestamp_ns": time.time_ns(),
                       "clock": "time.time_ns (epoch)",
                       "note": "main-table frame 0 is captured at/after this "
                               "instant; raw rows carry the same clock in "
                               "calibrated_timestamp_ns"}, f, indent=2)

    def write(self, finger: int, t_ns: int,
              resultant, taxels) -> None:
        f = self._files[finger]
        row = [str(t_ns), "0", "0", str(t_ns)]
        row += [f"{v:.1f}" for v in resultant]
        for p in taxels:
            row += [f"{p[0]:.1f}", f"{p[1]:.1f}", f"{p[2]:.1f}"]
        f.write(",".join(row) + "\n")
        f.flush()

    def close(self, discard: bool = False) -> None:
        import os, shutil
        for f in self._files:
            try:
                f.close()
            except Exception:
                pass
        self._files = []
        if discard:
            shutil.rmtree(self.dir, ignore_errors=True)


class PaxiniSensor(Sensor):
    """Driver for a Paxini PX-6AX GEN3 tactile sensor via paxini-sdk.

    Supports both the High-Speed Communication Board and the single-channel
    Serial Converter Board (``config.board_type``). Multiple instances on the
    same High-Speed board port share one board + stream (see
    ``_SharedHighSpeedBoard``).
    """

    def __init__(self, config: PaxiniSensorConfig):
        super().__init__(config)
        self.config: PaxiniSensorConfig = config

        # Lazy import so paxini-sdk is only required if this class is used.
        try:
            import paxini_sdk  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "paxini-sdk is not installed. Install it with: pip install paxini-sdk"
            ) from e

        self._board_type = getattr(config, "board_type", "high_speed")
        if self._board_type not in ("high_speed", "serial"):
            raise ValueError(
                f"Unsupported board_type: {self._board_type!r} "
                "(expected 'high_speed' or 'serial')"
            )

        self._board = None              # SingleSensorBoard (serial only)
        self._shared: Optional[_SharedHighSpeedBoard] = None  # high_speed only
        self._reader_attached = False
        self._is_connected = False
        self._stop_event = threading.Event()
        self._data_thread: threading.Thread | None = None   # serial only
        self._data_lock = threading.Lock()
        self._ring: Deque[np.ndarray] = deque(maxlen=config.buffer_size)
        self._sample_dim = None         # int, or (P, 3) tuple for distributed
        self._active_module_idx: int | None = None   # high-speed only
        self._module_name: str | None = None         # high-speed module / serial label
        self._n_taxels: int | None = None

        # High-speed downsampling state (set in start_continuous_read).
        self._period = 0.0
        self._last_ingest = 0.0

        # ---- combined multi-finger mode (company format) ----
        self._combined = config.output_format == "combined"
        self._module_names: list[str] = []          # per finger, list order
        self._latest_fingers: list[np.ndarray] | None = None
        self._raw_writer: "_RawCsvWriter | None" = None
        self._raw_lock = threading.Lock()

        # Optional software baseline (atop the firmware calibration)
        self._sw_baseline: Optional[np.ndarray] = None
        self._sw_baseline_buf: list[np.ndarray] = []

        # Cached anatomy coords for the optional rerun display path
        self._rerun_coords: Optional[list] = None
        self._rerun_anatomy_logged: bool = False
        self._rerun_log_path: Optional[str] = None

        # The dict key from --robot.sensors={NAME: ...}, set by the factory in
        # make_sensors_from_configs(). Used to log under
        # `observation.sensors.{name}/...` so the 3D point cloud appears in
        # lerobot's auto-built rerun Blueprint panel for this sensor.
        self._sensor_name: Optional[str] = None

    def set_sensor_name(self, name: str) -> None:
        """Called by ``make_sensors_from_configs`` so the sensor knows the
        dict key it lives under (e.g. ``paxini_fingertip``). Must be called
        before ``connect()`` for the rerun log path to use the correct
        Blueprint namespace."""
        self._sensor_name = name

    # ---- properties --------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def shape(self) -> tuple[int, ...]:
        if self._sample_dim is None:
            # Best-effort before connect(); finalized in connect().
            return (self.config.buffer_size, 3)
        if self._combined:
            # Combined mode has NO history buffer: one latest sample of
            # shape (n_fingers, P, 3) per observation (company format).
            return tuple(self._sample_dim)  # type: ignore[arg-type]
        if isinstance(self._sample_dim, tuple):
            return (self.config.buffer_size, *self._sample_dim)
        return (self.config.buffer_size, self._sample_dim)

    @property
    def is_calibrated(self) -> bool:
        if self.config.software_baseline_frames <= 0:
            return self._is_connected
        return self._sw_baseline is not None

    # ---- whether the read loop needs each data type ------------------------

    @property
    def _needs_resultant(self) -> bool:
        return self.config.output_format in ("resultant", "both")

    @property
    def _needs_distributed(self) -> bool:
        # display_rerun needs the per-taxel grid for the 3D point cloud even
        # if the recorded output is resultant-only.
        return (
            self.config.output_format in ("distributed", "both")
            or self.config.display_rerun
        )

    # ---- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self._board_type == "serial":
            self._connect_serial()
        else:
            self._connect_high_speed()

        # Resolve the recorded-sample dimensionality from the point count.
        if self._combined:
            self._sample_dim = (len(self._module_names), self._n_taxels, 3)  # type: ignore[assignment]
        elif self.config.output_format == "resultant":
            self._sample_dim = 3
        elif self.config.output_format == "distributed":
            self._sample_dim = (self._n_taxels, 3)  # type: ignore[assignment]
        elif self.config.output_format == "both":
            self._sample_dim = 3 + self._n_taxels * 3
        elif not self._combined:
            raise ValueError(
                f"Unsupported output_format: {self.config.output_format!r}"
            )

        self._is_connected = True
        logging.info(
            f"Connected to Paxini sensor on {self.config.port} "
            f"(board={self._board_type}, label={self._module_name}, "
            f"taxels={self._n_taxels}, output={self.config.output_format})"
        )
        self._preload_rerun_coords()

    def _connect_high_speed(self) -> None:
        from paxini_sdk import registers

        if self._combined:
            self._connect_combined()
            return

        # Acquire (or create) the shared board for this port. The first sensor
        # on the port opens it and reads the active-modules list; later
        # sensors reuse it.
        self._shared = _SharedHighSpeedBoard.acquire(
            self.config.port, self.config.baud_rate
        )
        try:
            active = self._shared.active_modules
            if not active:
                raise RuntimeError(
                    "No active Paxini modules detected. Check the FPC cable "
                    "orientation."
                )

            if self.config.module_index is None:
                self._active_module_idx = active[0]
            else:
                if self.config.module_index not in active:
                    raise RuntimeError(
                        f"Requested module {self.config.module_index} "
                        f"({registers.MODULE_NAMES[self.config.module_index]}) "
                        f"not active; active list is {active}."
                    )
                self._active_module_idx = self.config.module_index

            self._module_name = registers.MODULE_NAMES[self._active_module_idx]
            self._shared.claim_module(
                self._active_module_idx, self._sensor_name or self._module_name
            )
            self._n_taxels = self._shared.point_count(self._active_module_idx)

            if self.config.auto_calibrate:
                logging.info("Paxini: triggering firmware calibration ...")
                self._shared.request_calibration()
        except Exception:
            # Don't leak a refcount if setup fails after acquire().
            self._shared.release()
            self._shared = None
            raise

    def _connect_combined(self) -> None:
        """Combined multi-finger mode: one sensor claims ALL modules in
        config.module_indices and emits (n_fingers, P, 3). Company-format
        semantics: each observation is the LATEST raw sample per finger."""
        from paxini_sdk import registers

        idxs = self.config.module_indices
        if not idxs:
            raise RuntimeError(
                "output_format='combined' requires module_indices, e.g. "
                "module_indices: [10, 18] (finger order = list order)."
            )
        self._shared = _SharedHighSpeedBoard.acquire(
            self.config.port, self.config.baud_rate
        )
        try:
            active = self._shared.active_modules
            missing = [m for m in idxs if m not in active]
            if missing:
                raise RuntimeError(
                    f"Modules {missing} not active (active list: {active}). "
                    "Check FPC cabling / slot assignment."
                )
            counts = {m: self._shared.point_count(m) for m in idxs}
            if len(set(counts.values())) != 1:
                raise RuntimeError(
                    f"Mixed taxel counts across fingers not supported: {counts}"
                )
            self._n_taxels = counts[idxs[0]]
            self._module_names = [registers.MODULE_NAMES[m] for m in idxs]
            self._module_name = "+".join(self._module_names)
            self._active_module_idx = idxs[0]
            self._latest_fingers = [
                np.zeros((self._n_taxels, 3), dtype=np.float32)
                for _ in idxs
            ]
            for m, nm in zip(idxs, self._module_names):
                self._shared.claim_module(m, f"{self._sensor_name or 'paxini'}[{nm}]")
            if self.config.auto_calibrate:
                logging.info("Paxini: triggering firmware calibration ...")
                self._shared.request_calibration()
        except Exception:
            self._shared.release()
            self._shared = None
            raise

    def _connect_serial(self) -> None:
        from paxini_sdk import SingleSensorBoard, sensor_registry

        part_code = getattr(self.config, "sensor_part_code", None)
        if not part_code:
            raise RuntimeError(
                "board_type='serial' requires sensor_part_code in the config "
                "(e.g. 'PXSR-STDDP03A'). The Serial Converter Board cannot "
                "report which sensor is attached."
            )
        variant = sensor_registry.find_by_part_code(part_code)
        if variant is None:
            known = [s.vendor_part_code for s in sensor_registry.list_sensors()]
            raise RuntimeError(
                f"Unknown sensor_part_code {part_code!r}. Known codes: {known}"
            )
        self._n_taxels = variant.n_points
        self._module_name = variant.vendor_part_code

        self._board = SingleSensorBoard(
            self.config.port,
            baudrate=self.config.baud_rate,
            device_id=getattr(self.config, "device_id", 0x01),
            sensor=variant,
        )
        self._board.open()

        if self.config.auto_calibrate:
            logging.info("Paxini: triggering Serial Converter Board calibration ...")
            self._board.calibrate()  # synchronous request/response

    def _preload_rerun_coords(self) -> None:
        """Load the sensor's anatomy coordinates for the optional 3D rerun
        point cloud. No-op if display_rerun is off or coords are unavailable."""
        if not self.config.display_rerun:
            return
        try:
            from paxini_sdk import sensor_registry
            variant = sensor_registry.find_by_point_count(self._n_taxels)
            if variant is None:
                logging.warning("Paxini: no registry match for "
                                f"{self._n_taxels} points; display_rerun is a no-op.")
                return
            self._rerun_coords = sensor_registry.load_points(variant)
            if self._sensor_name:
                # Slash-separated so the entity tree is a real hierarchy
                # (/observation/sensors/<name>/...). rerun splits paths on "/"
                # only, so a dotted "observation.sensors.<name>" would be ONE
                # opaque part that /observation/sensors/** can't match. See the
                # default blueprint in visualization_utils.py.
                self._rerun_log_path = f"observation/sensors/{self._sensor_name}"
            else:
                safe = (self._module_name or "paxini").lower().replace("-", "_")
                self._rerun_log_path = f"sensor/{safe}"
            logging.info(
                f"Paxini: rerun 3D point cloud enabled for "
                f"{variant.vendor_part_code} at '{self._rerun_log_path}'"
            )
        except Exception as e:
            logging.warning(f"Paxini: rerun coords unavailable ({e}); "
                            "the display_rerun flag is a no-op.")

    def disconnect(self) -> None:
        self.finish_raw_episode()
        self.stop_continuous_read()
        if self._board_type == "serial":
            if self._board is not None:
                try:
                    self._board.close()
                except Exception:
                    pass
                self._board = None
        else:
            if self._shared is not None:
                self._shared.release()
                self._shared = None
        self._is_connected = False
        logging.info("Disconnected from Paxini sensor")

    # ---- threading ---------------------------------------------------------

    def start_continuous_read(self) -> None:
        if not self._is_connected:
            raise RuntimeError("Sensor must be connected before start_continuous_read()")
        self._stop_event.clear()

        if self._board_type == "serial":
            if self._data_thread and self._data_thread.is_alive():
                return
            self._data_thread = threading.Thread(
                target=self._continuous_read_loop, daemon=True
            )
            self._data_thread.start()
        else:
            # High-speed: attach this sensor's per-frame callback to the shared
            # stream and make sure the shared stream thread is running.
            rate = max(0.0, float(getattr(self.config, "poll_rate_hz", 0.0)))
            self._period = (1.0 / rate) if rate > 0 else 0.0
            self._last_ingest = 0.0
            if self._shared is not None:
                self._shared.attach_reader(self._on_frame)
                self._reader_attached = True
                self._shared.ensure_stream_started()
        logging.info("Started Paxini sensor continuous read")

    def stop_continuous_read(self) -> None:
        self._stop_event.set()
        if self._board_type == "serial":
            if self._data_thread and self._data_thread.is_alive():
                self._data_thread.join(timeout=2.0)
            self._data_thread = None
        else:
            if self._reader_attached and self._shared is not None:
                self._shared.detach_reader(self._on_frame)
                self._reader_attached = False

    def _on_frame(self, frame) -> None:
        """Per-frame callback invoked by the shared board's stream thread.
        Applies this sensor's poll-rate downsampling, extracts this sensor's
        module from the shared frame, and ingests it."""
        if self._stop_event.is_set():
            return
        if self._combined:
            # Company-format mode: keep the latest (P, 3) grid per finger at
            # the FULL stream rate (no downsampling — get_latest_data slices
            # at observation time), and mirror every frame to the raw CSVs.
            t_ns = time.time_ns()
            for k, name in enumerate(self._module_names):
                pts = frame.distributed_forces_newtons.get(name)
                if pts:
                    arr = np.asarray(pts, dtype=np.float32)
                    if arr.shape[0] < self._n_taxels:
                        arr = np.concatenate(
                            [arr, np.zeros((self._n_taxels - arr.shape[0], 3),
                                            dtype=np.float32)], axis=0)
                    with self._data_lock:
                        self._latest_fingers[k] = arr[: self._n_taxels]
                res = frame.resultant_forces_newtons.get(name)
                with self._raw_lock:
                    if self._raw_writer is not None and pts:
                        self._raw_writer.write(
                            k, t_ns, res or (0.0, 0.0, 0.0),
                            self._latest_fingers[k])
                if self.config.display_rerun and k == 0:
                    self._log_to_rerun(res, pts)
            return
        if self._period:
            now = time.monotonic()
            if now - self._last_ingest < self._period:
                return  # drop; keep the ring at the target rate
            self._last_ingest = now
        res_n = frame.resultant_forces_newtons.get(self._module_name)
        dist_n = frame.distributed_forces_newtons.get(self._module_name) or None
        self._ingest(res_n, dist_n)

    def _continuous_read_loop(self) -> None:
        # Serial board only (high-speed uses the shared stream + _on_frame).
        try:
            self._serial_read_loop()
        except Exception as e:
            logging.exception(f"Paxini read loop terminated: {e}")

    def _serial_read_loop(self) -> None:
        """Poll the Serial Converter Board (request/response).

        The board has no auto-push. With the length-based read in
        paxini_sdk.transport.fin_transaction each round-trip is fast, so the
        loop can run well above the old ~8-16 Hz. If config.poll_rate_hz > 0
        the loop is locked to that rate by sleeping the remainder of each
        1/rate period -- a stable, deterministic cadence like the Teensy
        gives the MLX hall sensor. poll_rate_hz = 0 polls as fast as the
        board answers. Failed transactions are logged once and retried."""
        rate = max(0.0, float(getattr(self.config, "poll_rate_hz", 0.0)))
        period = (1.0 / rate) if rate > 0 else 0.0
        warned = False
        rate_warned = False
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                res_n: _ResN = None
                dist_n: _DistN = None
                if self._needs_resultant:
                    fx, fy, fz = self._board.read_resultant_force()
                    res_n = (fx * _SERIAL_LSB_N, fy * _SERIAL_LSB_N,
                             fz * _SERIAL_LSB_N)
                if self._needs_distributed:
                    pts = self._board.read_distributed_force()
                    dist_n = [
                        (p.fx * _SERIAL_LSB_N, p.fy * _SERIAL_LSB_N,
                         p.fz * _SERIAL_LSB_N)
                        for p in pts
                    ]
                self._ingest(res_n, dist_n)
            except Exception as e:
                if not warned:
                    logging.warning(f"Paxini serial read error (retrying): {e}")
                    warned = True
                self._stop_event.wait(0.1)
                continue
            # Rate lock: sleep whatever is left of this period. Uses the stop
            # event so a disconnect interrupts the wait promptly.
            if period:
                remaining = period - (time.monotonic() - cycle_start)
                if remaining > 0:
                    self._stop_event.wait(remaining)
                elif not rate_warned:
                    logging.warning(
                        f"Paxini: serial board can't sustain poll_rate_hz="
                        f"{rate:.0f} (a read cycle took longer than "
                        f"{period * 1000:.0f} ms); running as fast as possible."
                    )
                    rate_warned = True

    # ---- shared sample ingestion ------------------------------------------

    def _ingest(self, res_n: _ResN, dist_n: _DistN) -> None:
        """Turn one (resultant, distributed) reading into a recorded sample:
        build the tensor, apply the optional software baseline, append to the
        ring buffer, and (optionally) log the 3D point cloud to rerun."""
        sample = self._build_sample(res_n, dist_n)
        if sample is None:
            return

        # Optional host-side baseline calibration (atop firmware calibration)
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
            return
        if self._sw_baseline is not None:
            sample = sample - self._sw_baseline

        with self._data_lock:
            self._ring.append(sample)

        if self.config.display_rerun:
            self._log_to_rerun(res_n, dist_n)

    def _build_sample(self, res_n: _ResN, dist_n: _DistN) -> np.ndarray | None:
        """Build a float32 sample of shape (3,), (P, 3), or (3 + P*3,)
        depending on output_format, from a (resultant, distributed) reading."""
        fmt = self.config.output_format
        if fmt == "resultant":
            if res_n is None:
                return None
            return np.array(res_n, dtype=np.float32)

        if fmt == "distributed":
            if not dist_n:
                return None
            arr = np.array(dist_n, dtype=np.float32)
            return self._pad_taxels(arr)

        if fmt == "both":
            if res_n is None and not dist_n:
                return None
            res_arr = np.array(res_n or (0.0, 0.0, 0.0), dtype=np.float32)
            pts_arr = np.array(
                dist_n or [(0.0, 0.0, 0.0)] * self._n_taxels, dtype=np.float32
            )
            pts_arr = self._pad_taxels(pts_arr)
            return np.concatenate([res_arr, pts_arr.flatten()], axis=0)
        return None

    def _pad_taxels(self, arr: np.ndarray) -> np.ndarray:
        """Pad/trim a (P', 3) array to exactly (n_taxels, 3)."""
        if arr.shape[0] < self._n_taxels:
            pad = np.zeros((self._n_taxels - arr.shape[0], 3), dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)
        return arr[: self._n_taxels]

    @staticmethod
    def _force_colormap(t: float):
        """green -> yellow -> orange -> red ramp for a normalized force t in
        [0, 1], matching the PXSR host app's magnitude scale."""
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        # stops: 0.0 green, 0.4 yellow, 0.7 orange, 1.0 red
        stops = [(0.0, (0, 180, 40)), (0.4, (235, 235, 0)),
                 (0.7, (245, 140, 0)), (1.0, (220, 0, 0))]
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                f = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return (round(c0[0] + (c1[0] - c0[0]) * f),
                        round(c0[1] + (c1[1] - c0[1]) * f),
                        round(c0[2] + (c1[2] - c0[2]) * f))
        return stops[-1][1]

    def _log_to_rerun(self, res_n: _ResN, dist_n: _DistN) -> None:
        """Draw each taxel's (Fx, Fy, Fz) force as a 3D arrow (PXSR-style):
        the arrow starts at the taxel position, points along the force vector,
        and is colored by magnitude on a green->red scale. A faint static
        point cloud shows the fingertip shape at rest. Plus per-axis resultant
        scalars. Safe no-op if rerun isn't running or coords weren't loaded."""
        if self._rerun_coords is None or self._rerun_log_path is None:
            return
        try:
            import rerun as rr
        except ImportError:
            return
        try:
            if not self._rerun_anatomy_logged:
                # Orient the view Y-up so the fingertip stands upright and
                # faces the camera, matching the PXSR host app (the sensor's
                # long axis is +Y, width is X, dome bulge is Z).
                rr.log(self._rerun_log_path,
                       rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
                # Faint backdrop so the fingertip shape is always visible,
                # even with no contact. Static so it's logged once.
                pale = [(150, 155, 165)] * len(self._rerun_coords)
                rr.log(
                    f"{self._rerun_log_path}/anatomy",
                    rr.Points3D(positions=self._rerun_coords,
                                 colors=pale, radii=0.12),
                    static=True,
                )
                self._rerun_anatomy_logged = True

            if dist_n:
                cfg = self.config
                scale = float(getattr(cfg, "rerun_arrow_scale_mm_per_n", 2.0))
                fmax = max(1e-6, float(getattr(cfg, "rerun_force_max_n", 10.0)))
                thr = float(getattr(cfg, "rerun_force_threshold_n", 0.15))
                n = min(len(dist_n), len(self._rerun_coords))
                origins, vectors, colors = [], [], []
                for i in range(n):
                    fx, fy, fz = dist_n[i]
                    mag = (fx * fx + fy * fy + fz * fz) ** 0.5
                    if mag < thr:
                        continue  # skip near-zero taxels (noise)
                    origins.append(self._rerun_coords[i])
                    vectors.append((fx * scale, fy * scale, fz * scale))
                    colors.append(self._force_colormap(mag / fmax))
                # Log even when empty so released taxels clear their arrows.
                rr.log(
                    f"{self._rerun_log_path}/force_arrows",
                    rr.Arrows3D(origins=origins, vectors=vectors,
                                 colors=colors, radii=0.15),
                )

            if res_n is not None:
                rr.log(f"{self._rerun_log_path}/resultant/fx", rr.Scalars(res_n[0]))
                rr.log(f"{self._rerun_log_path}/resultant/fy", rr.Scalars(res_n[1]))
                rr.log(f"{self._rerun_log_path}/resultant/fz", rr.Scalars(res_n[2]))
        except Exception:
            pass

    # ---- data access -------------------------------------------------------

    def get_latest_data(self) -> np.ndarray | None:
        if not self._is_connected:
            return None
        if not self.is_calibrated:
            return None
        if self._combined:
            with self._data_lock:
                return np.stack(self._latest_fingers).astype(np.float32)
        with self._data_lock:
            samples = list(self._ring)
        if not samples:
            return np.zeros(self.shape, dtype=np.float32)

        if self.config.output_format == "distributed":
            # Each sample is (P, 3); stack -> (N, P, 3)
            if len(samples) < self.config.buffer_size:
                pad = np.zeros(
                    (self.config.buffer_size - len(samples), self._n_taxels, 3),
                    dtype=np.float32,
                )
                return np.concatenate([pad, np.stack(samples)], axis=0)
            return np.stack(samples)

        # Resultant / both: each sample is (D,)
        if len(samples) < self.config.buffer_size:
            pad = np.zeros(
                (self.config.buffer_size - len(samples), self._sample_dim),
                dtype=np.float32,
            )
            return np.concatenate([pad, np.stack(samples)], axis=0)
        return np.stack(samples)

    # ---- raw-CSV episode lifecycle (called by lerobot_record) --------------

    def start_raw_episode(self, dataset_root, episode_index: int) -> None:
        """Open per-finger raw CSVs for the episode about to be recorded.
        No-op unless output_format='combined' and record_raw_csv=True."""
        if not (self._combined and getattr(self.config, "record_raw_csv", False)):
            return
        with self._raw_lock:
            if self._raw_writer is not None:
                self._raw_writer.close()
            self._raw_writer = _RawCsvWriter(
                str(dataset_root), self._sensor_name or "paxini",
                episode_index, len(self._module_names), self._n_taxels,
                getattr(self.config, "raw_csv_filename", "sensor_{n}.csv"),
            )
        logging.info(
            f"Paxini: raw CSV sidecar -> {self._raw_writer.dir}"
        )

    def finish_raw_episode(self) -> None:
        with self._raw_lock:
            if self._raw_writer is not None:
                self._raw_writer.close()
                self._raw_writer = None

    def discard_raw_episode(self) -> None:
        """Re-record: drop the episode's raw files entirely."""
        with self._raw_lock:
            if self._raw_writer is not None:
                self._raw_writer.close(discard=True)
                self._raw_writer = None

    def wait_for_calibration(self, timeout_s: float = 10.0, poll_s: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self.is_calibrated:
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_s)
        return True

    def metadata(self) -> dict:
        return {
            **super().metadata(),
            "board_type": self._board_type,
            "port": self.config.port,
            "baud_rate": self.config.baud_rate,
            "buffer_size": self.config.buffer_size,
            "module_index": self._active_module_idx,
            "module_name": self._module_name,
            "sensor_part_code": getattr(self.config, "sensor_part_code", None),
            "device_id": getattr(self.config, "device_id", None),
            "n_taxels": self._n_taxels,
            "output_format": self.config.output_format,
            "auto_calibrate": self.config.auto_calibrate,
            "software_baseline_frames": self.config.software_baseline_frames,
            "is_calibrated": self.is_calibrated,
            "sw_baseline": (
                self._sw_baseline.tolist() if self._sw_baseline is not None else None
            ),
        }
