# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numbers
import os

import numpy as np

from lerobot.types import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
from .import_utils import require_package


_SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _default_so101_blueprint():
    """Build the default Cowork/SO-101 layout used by ``init_rerun``.

    Three rows x two columns:
      (1) Paxini force time series  |  Paxini 3D point cloud
      (2) Wrist camera              |  Top camera
      (3) Joint positions (all 12)  |  Hall sensor (MLX) 10x3 buffer

    Missing entities render as empty panels - safe to leave in even when
    the user runs without one of the sensors.
    """
    import rerun.blueprint as rrb

    paxini_root = "/observation.sensors.paxini_fingertip"

    paxini_forces = rrb.TimeSeriesView(
        name="Paxini forces (N)",
        contents=[
            f"{paxini_root}/sum_fx",
            f"{paxini_root}/sum_fy",
            f"{paxini_root}/sum_fz",
            f"{paxini_root}/resultant/**",
        ],
    )
    paxini_3d = rrb.Spatial3DView(
        name="Paxini fingertip",
        origin=paxini_root,
        contents=[f"{paxini_root}/anatomy", f"{paxini_root}/distributed"],
    )

    wrist_cam = rrb.Spatial2DView(name="Wrist", origin="/observation.wrist")
    top_cam = rrb.Spatial2DView(name="Top", origin="/observation.top")

    joint_contents = [f"/action.{j}.pos" for j in _SO101_JOINTS] + [
        f"/observation.{j}.pos" for j in _SO101_JOINTS
    ]
    joints = rrb.TimeSeriesView(name="Joint positions", contents=joint_contents)

    # MLX gripper buffer as a (10, 3) heatmap; updated per tick by the
    # ndim==2 branch of `log_rerun_data` below.
    hall = rrb.Spatial2DView(
        name="Hall sensor (10x3)",
        origin="/observation.sensors.gripper/array",
        contents=["/observation.sensors.gripper/array"],
    )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(paxini_forces, paxini_3d),
            rrb.Horizontal(wrist_cam, top_cam),
            rrb.Horizontal(joints, hall),
        ),
        collapse_panels=True,
    )


def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
    else:
        rr.spawn(memory_limit=memory_limit)

    # Push a default blueprint matching the SO-101 + Paxini fingertip
    # workflow. Set LEROBOT_DEFAULT_BLUEPRINT=0 to skip and fall back to
    # rerun's auto-Blueprint.
    bp_env = os.getenv("LEROBOT_DEFAULT_BLUEPRINT", "1")
    if bp_env not in ("0", "false", "False"):
        try:
            rr.send_blueprint(_default_so101_blueprint(), make_active=True, make_default=True)
        except Exception as e:
            import logging
            logging.warning(f"Skipping default lerobot blueprint: {e}")


def shutdown_rerun() -> None:
    """Shuts down the Rerun SDK gracefully."""

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    rr.rerun_shutdown()


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """

    require_package("rerun-sdk", extra="viz", import_name="rerun")
    import rerun as rr

    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                elif arr.ndim == 2 and arr.shape[-1] <= 16:
                    # Sensor ring buffer of shape (N, D). Log TWO things:
                    #   - `{key}/array`: the full (N, D) buffer as a small
                    #     heatmap image. Renders as e.g. a 10x3 strip for
                    #     MLX, giving the user the full ring-buffer state
                    #     at every tick instead of only the latest sample.
                    #   - `{key}_0`, `{key}_1`, ...: the latest sample's
                    #     D channels as scalars, for time-series plotting.
                    rr.log(f"{key}/array", rr.Image(arr.astype(np.float32)))
                    latest = arr[-1]
                    for i, vi in enumerate(latest):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                elif (
                    arr.ndim == 3
                    and arr.shape[-1] == 3
                    and arr.shape[0] <= 16
                    and arr.shape[1] <= 256
                ):
                    # Tactile sensor ring buffer of shape (N, P, 3).
                    # N is the buffer_size (small, <= 16); P is the taxel
                    # count. Camera frames are also (H, W, 3) and ndim==3,
                    # so the size guards are essential - otherwise this
                    # branch hijacks every camera log. P caps at ~100 across
                    # all Paxini variants in the registry; 256 is generous.
                    # We only log per-axis sums here; the spatial layout is
                    # better served by the 3D point cloud that
                    # PaxiniSensor._log_to_rerun emits under
                    # `observation.sensors.{name}/distributed` when
                    # `display_rerun: true`.
                    latest = arr[-1]
                    rr.log(f"{key}/sum_fx", rr.Scalars(float(latest[:, 0].sum())))
                    rr.log(f"{key}/sum_fy", rr.Scalars(float(latest[:, 1].sum())))
                    rr.log(f"{key}/sum_fz", rr.Scalars(float(latest[:, 2].sum())))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
