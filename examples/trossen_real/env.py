"""Robot environment for deploying the torque-aware pi0 (TA-VLA) on the real Trossen
bimanual WidowX AI follower arms.

This is the client-side counterpart to openpi's websocket policy server. It mirrors
``examples/aloha_real/env.py`` but talks to the ``lerobot_robot_trossen`` follower and, in
addition to the state and camera observations, maintains the rolling **effort-history buffer**
the TA-VLA model consumes (``EXPERT_HIS_C_FUT``).

It intentionally imports only ``openpi_client`` (not the JAX ``openpi`` package) plus
``lerobot``/``lerobot_robot_trossen``, so it runs in the ``~/lerobot_trossen/.venv``.

Observation contract sent to the server (``TavlaInputs`` form):
- ``state``:  float32[14] = ``[L joint_0..5, L carriage, R joint_0..5, R carriage]`` (rad; gripper m)
- ``effort``: float32[10, 14] = 10 history frames of ``[L ext_eff(7), R ext_eff(7)]`` (Nm/N),
              ordered oldest->newest at frame offsets ``(-36, -32, ..., 0)``
- ``images``: ``{cam_high, cam_left_wrist, cam_right_wrist}`` raw HWC uint8 (server resizes to 224)
- ``prompt``: the exact training task string

The server returns ``{"actions": np[H, 14]}`` already **absolute** (delta/normalization are all
server-side); ``ActionChunkBroker`` hands us one 14-vector per tick, which we map back onto the
follower's ``.pos`` action keys.
"""

from __future__ import annotations

import collections
import logging
import os
from typing import Optional  # noqa: UP035

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_trossen.config_bi_widowxai_follower import (
    BiWidowXAIFollowerRobotConfig,
)

logger = logging.getLogger(__name__)

# Per-arm joint order used by the WidowX AI follower (see config_widowxai_follower.py).
# The gripper carriage is literally named "left_carriage_joint" on both arms, so with the
# bimanual "left_"/"right_" prefix it becomes "left_left_carriage_joint" / "right_left_carriage_joint".
_JOINT_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "left_carriage_joint",
)

# 14-dim vectors, left arm first, matching the training dataset split
# (state[[0:7]+[14:21]] positions, state[[7:14]+[21:28]] external efforts).
STATE_KEYS = tuple(f"{side}_{j}.pos" for side in ("left", "right") for j in _JOINT_NAMES)
EFFORT_KEYS = tuple(f"{side}_{j}.ext_eff" for side in ("left", "right") for j in _JOINT_NAMES)
ACTION_KEYS = STATE_KEYS  # action is 14 absolute joint positions in the same order

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# Effort-history frame offsets the SOTA config trains with: (-36, -32, ..., 0), oldest->newest.
# Must byte-match `effort_history` in `pi0_trossen_transfer_effort_sota`.
EFFORT_HISTORY_OFFSETS = tuple(4 * i - 36 for i in range(10))

# No default prompt on purpose: the prompt must match the served policy's training string, and a
# stale default is silent (the server only injects its own prompt when the key is absent, so a
# client-side prompt always wins). `main.py` resolves it from the server's metadata instead.

# Start pose the arms are staged to on connect (rad for the 6 arm joints, m for the gripper
# carriage). All-zeros matches the pose the training dataset was recorded from, so each episode
# begins in-distribution. Order: [joint_0..5, left_carriage_joint].
STAGED_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TrossenRealEnvironment(_environment.Environment):
    """openpi ``Environment`` backed by the Trossen bimanual WidowX AI follower."""

    def __init__(
        self,
        *,
        left_arm_ip: Optional[str] = None,  # noqa: UP007
        right_arm_ip: Optional[str] = None,  # noqa: UP007
        cam_high_serial: Optional[str] = None,  # noqa: UP007
        cam_left_wrist_serial: Optional[str] = None,  # noqa: UP007
        cam_right_wrist_serial: Optional[str] = None,  # noqa: UP007
        prompt: str,
        effort_history_offsets: tuple[int, ...] = EFFORT_HISTORY_OFFSETS,
        max_relative_target: float | None = 1.0,
        action_ema_alpha: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        self._prompt = prompt
        self._dry_run = dry_run

        # oldest->newest ordering; index for frame offset `o` (<= 0) is deque[o - 1].
        self._offsets = tuple(sorted(effort_history_offsets))
        if self._offsets[-1] != 0:
            raise ValueError(f"effort_history_offsets must end at 0 (newest); got {self._offsets}")
        self._deque_maxlen = 1 - min(self._offsets)  # e.g. offsets down to -36 -> maxlen 37
        self._effort_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=self._deque_maxlen)

        if not (0.0 < action_ema_alpha <= 1.0):
            raise ValueError(f"action_ema_alpha must lie in (0, 1]; got {action_ema_alpha}")
        self._action_ema_alpha = action_ema_alpha
        self._ema_state: np.ndarray | None = None

        left_arm_ip = left_arm_ip or os.environ["FOLLOWER_LEFT_IP_ADDR"]
        right_arm_ip = right_arm_ip or os.environ["FOLLOWER_RIGHT_IP_ADDR"]
        cam_high_serial = cam_high_serial or os.environ.get("CAM_HIGH_SN", "419122270126")
        cam_left_wrist_serial = cam_left_wrist_serial or os.environ.get("CAM_LEFT_WRIST_SN", "412622272448")
        cam_right_wrist_serial = cam_right_wrist_serial or os.environ.get("CAM_RIGHT_WRIST_SN", "412622272396")

        def _cam(serial: str) -> RealSenseCameraConfig:
            return RealSenseCameraConfig(serial_number_or_name=serial, fps=30, width=640, height=480)

        config = BiWidowXAIFollowerRobotConfig(
            left_arm_ip_address=left_arm_ip,
            right_arm_ip_address=right_arm_ip,
            include_external_effort=True,
            left_arm_max_relative_target=max_relative_target,
            right_arm_max_relative_target=max_relative_target,
            # Start each episode in-distribution at the data-collection pose (all-zeros).
            left_arm_staged_positions=list(STAGED_POSITIONS),
            right_arm_staged_positions=list(STAGED_POSITIONS),
            cameras={
                "cam_high": _cam(cam_high_serial),
                "cam_left_wrist": _cam(cam_left_wrist_serial),
                "cam_right_wrist": _cam(cam_right_wrist_serial),
            },
        )

        logger.info(
            "Connecting Trossen bimanual follower (left=%s right=%s, dry_run=%s)...",
            left_arm_ip,
            right_arm_ip,
            dry_run,
        )
        self._robot = make_robot_from_config(config)
        # connect() drives both arms to their staged start pose (all-zeros) and opens the cameras.
        self._robot.connect()
        logger.info("Follower connected. Cameras: %s", CAMERA_KEYS)

    @override
    def reset(self) -> None:
        # Re-seed the effort buffer from the current reading so history is defined at step 0.
        self._ema_state = None
        obs = self._robot.get_observation()
        effort = self._extract_effort(obs)
        self._effort_buffer.clear()
        for _ in range(self._deque_maxlen):
            self._effort_buffer.append(effort.copy())

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        obs = self._robot.get_observation()

        state = np.array([float(obs[k]) for k in STATE_KEYS], dtype=np.float32)
        effort = self._extract_effort(obs)
        self._effort_buffer.append(effort)

        # oldest->newest history at the trained offsets. deque[o - 1] selects frame offset o (<= 0).
        effort_history = np.stack([self._effort_buffer[o - 1] for o in self._offsets], axis=0).astype(np.float32)

        images = {}
        for cam in CAMERA_KEYS:
            img = np.asarray(obs[cam])
            images[cam] = image_tools.convert_to_uint8(img)

        return {
            "state": state,
            "effort": effort_history,
            "images": images,
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        actions = np.asarray(action["actions"], dtype=np.float32).reshape(-1)[:14]

        if self._action_ema_alpha < 1.0:
            if self._ema_state is None:
                self._ema_state = actions.copy()
            else:
                self._ema_state = self._action_ema_alpha * actions + (1.0 - self._action_ema_alpha) * self._ema_state
            actions = self._ema_state

        if self._dry_run:
            logger.info("[dry-run] action (not sent): %s", np.array2string(actions, precision=4))
            return

        action_dict = {key: float(actions[i]) for i, key in enumerate(ACTION_KEYS)}
        self._robot.send_action(action_dict)

    def close(self) -> None:
        """Drive the arms to the staged/sleep pose and release the hardware."""
        try:
            self._robot.disconnect()
        except Exception:  # noqa: BLE001 - best-effort cleanup on shutdown
            logger.exception("Error while disconnecting the follower.")

    @staticmethod
    def _extract_effort(obs: dict) -> np.ndarray:
        return np.array([float(obs[k]) for k in EFFORT_KEYS], dtype=np.float32)
