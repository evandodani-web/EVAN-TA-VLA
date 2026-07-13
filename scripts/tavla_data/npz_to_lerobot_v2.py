#!/usr/bin/env python
"""Step 2/2 of the TA-VLA data conversion: build a lerobot **v2.x** dataset from the
per-episode ``.npz`` intermediate produced by ``export_v30_to_npz.py``.

Run this with the openpi venv (this repo's ``.venv``, lerobot 0.1.0):

    /home/trossen/EVAN-TA-VLA/.venv/bin/python \
        scripts/tavla_data/npz_to_lerobot_v2.py \
        --intermediate /home/trossen/tavla_intermediate \
        --repo-id trossen_bimanual_transfer_cube_tavla

Output goes to ``$LEROBOT_HOME/<repo-id>`` (``~/.cache/huggingface/lerobot/<repo-id>``)
unless ``--root`` is given. The resulting dataset has:

    observation.state           (14,)  joint positions  [L pos, R pos]
    observation.effort          (14,)  external effort   [L ext_eff, R ext_eff]
    action                      (14,)  joint positions
    observation.images.cam_high         (video)
    observation.images.cam_left_wrist   (video)
    observation.images.cam_right_wrist  (video)

This is exactly what ``LeRobotTavlaDataConfig`` / ``TavlaInputs`` expect. It is NOT pushed
to the Hub; rsync it to the training box afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import cv2
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset

# Left-first joint names, matching the source dataset's meta/info.json.
_JOINTS = [
    "left_joint_0", "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4",
    "left_joint_5", "left_left_carriage_joint",
    "right_joint_0", "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4",
    "right_joint_5", "right_left_carriage_joint",
]
POS_NAMES = [f"{j}.pos" for j in _JOINTS]
EFF_NAMES = [f"{j}.ext_eff" for j in _JOINTS]

DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
IMG_H, IMG_W = 480, 640


def _build_features(cameras: tuple[str, ...], mode: str) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": POS_NAMES},
        "observation.effort": {"dtype": "float32", "shape": (14,), "names": EFF_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": POS_NAMES},
    }
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, IMG_H, IMG_W),
            "names": ["channels", "height", "width"],
        }
    return features


def _decode_jpeg(jpeg_uint8: np.ndarray) -> np.ndarray:
    """JPEG bytes -> HWC RGB uint8."""
    bgr = cv2.imdecode(np.asarray(jpeg_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("cv2.imdecode failed on a stored frame")
    return np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--intermediate", required=True, type=Path, help="Dir with episode_*.npz files.")
    ap.add_argument("--repo-id", default="trossen_bimanual_transfer_cube_tavla")
    ap.add_argument("--root", type=Path, default=None, help="Output root (default: $LEROBOT_HOME/<repo-id>).")
    ap.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mode", choices=["video", "image"], default="video")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    npz_files = sorted(args.intermediate.glob("episode_*.npz"))
    if not npz_files:
        raise SystemExit(f"No episode_*.npz found in {args.intermediate}")
    cameras = tuple(args.cameras)

    out_root = args.root if args.root is not None else (LEROBOT_HOME / args.repo_id)
    if out_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists (use --overwrite): {out_root}")
        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=str(args.root) if args.root is not None else None,
        robot_type="bi_widowxai_follower_robot",
        features=_build_features(cameras, args.mode),
        use_videos=(args.mode == "video"),
        image_writer_processes=0,
        image_writer_threads=4,
    )

    print(f"Building {args.repo_id} from {len(npz_files)} episodes -> {dataset.root}")
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        qpos = data["qpos"].astype(np.float32)
        effort = data["effort"].astype(np.float32)
        action = data["action"].astype(np.float32)
        task = str(data["task"])
        num_frames = qpos.shape[0]
        imgs = {cam: data[f"image.{cam}"] for cam in cameras}

        for i in range(num_frames):
            frame = {
                "observation.state": torch.from_numpy(qpos[i]),
                "observation.effort": torch.from_numpy(effort[i]),
                "action": torch.from_numpy(action[i]),
            }
            for cam in cameras:
                frame[f"observation.images.{cam}"] = _decode_jpeg(imgs[cam][i])
            dataset.add_frame(frame)
        dataset.save_episode(task=task)
        print(f"  {npz_path.name}: {num_frames} frames  task={task!r}")

    dataset.consolidate()
    print(f"Done. v2.x dataset at: {dataset.root}")
    print("Sanity-check with scripts/compute_norm_stats.py after wiring a train config.")


if __name__ == "__main__":
    main()
