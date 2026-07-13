#!/usr/bin/env python
"""Step 1/2 of the TA-VLA data conversion: export a lerobot **v3.0** dataset to a
version-agnostic per-episode ``.npz`` intermediate.

Run this with the ``lerobot_trossen`` venv (lerobot >= 0.5, which can read v3.0):

    /home/trossen/lerobot_trossen/.venv/bin/python \
        scripts/tavla_data/export_v30_to_npz.py \
        --src /home/trossen/.cache/huggingface/lerobot/datasets/trossen-bimanual-transfer-cube-external-effort-v2 \
        --out /home/trossen/tavla_intermediate

What it does
------------
The source ``observation.state`` is 28-dim and interleaves, per arm, 7 joint positions
then 7 external-effort (``.ext_eff``) values::

    [0:7]   left  *.pos
    [7:14]  left  *.ext_eff
    [14:21] right *.pos
    [21:28] right *.ext_eff

TA-VLA wants a 14-dim position-only ``observation.state`` plus a *separate* 14-dim
``observation.effort`` column (so effort history/future can be sampled). This script splits
them:

    qpos   = state[[0:7] + [14:21]]   (14, left-first, matches ``action``)
    effort = state[[7:14] + [21:28]]  (14, left-first)

Images are decoded (AV1 -> RGB) by lerobot and re-stored as JPEG bytes to keep the
intermediate small. Step 2 (``npz_to_lerobot_v2.py``, run in the openpi venv) turns these
npz files into a v2.x LeRobotDataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# State split (left-first), matching meta/info.json of the source dataset.
POS_IDX = list(range(0, 7)) + list(range(14, 21))
EFF_IDX = list(range(7, 14)) + list(range(21, 28))

DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


def _chw_float_to_hwc_uint8(img) -> np.ndarray:
    """lerobot returns images as CHW float32 RGB in [0, 1]; convert to HWC uint8 RGB."""
    arr = np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D image tensor, got shape {arr.shape}")
    # CHW -> HWC
    if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _encode_jpeg(hwc_rgb_uint8: np.ndarray, quality: int) -> np.ndarray:
    bgr = hwc_rgb_uint8[:, :, ::-1]  # RGB -> BGR for cv2
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.reshape(-1)  # 1-D uint8


def _flush_episode(out_dir: Path, ep_idx: int, buf: dict, cameras: tuple[str, ...]) -> None:
    qpos = np.asarray(buf["qpos"], dtype=np.float32)
    effort = np.asarray(buf["effort"], dtype=np.float32)
    action = np.asarray(buf["action"], dtype=np.float32)
    payload: dict[str, object] = {
        "qpos": qpos,
        "effort": effort,
        "action": action,
        "task": np.array(buf["task"]),
        "fps": np.array(buf["fps"], dtype=np.float32),
    }
    for cam in cameras:
        frames = buf["images"][cam]
        obj = np.empty(len(frames), dtype=object)
        for i, f in enumerate(frames):
            obj[i] = f
        payload[f"image.{cam}"] = obj
    out_path = out_dir / f"episode_{ep_idx:06d}.npz"
    np.savez(out_path, **payload)
    print(f"  wrote {out_path.name}  frames={len(action)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="Path to the v3.0 lerobot dataset root.")
    ap.add_argument("--out", required=True, type=Path, help="Output dir for episode_*.npz files.")
    ap.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--max-episodes", type=int, default=None, help="Only export the first N episodes (debug).")
    args = ap.parse_args()

    src: Path = args.src
    if not src.exists():
        raise SystemExit(f"Source dataset not found: {src}")
    args.out.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(repo_id=src.name, root=str(src))
    print(f"Loaded {src.name}: frames={ds.num_frames} episodes={ds.num_episodes} fps={ds.fps}")
    print(f"Cameras: {args.cameras} | JPEG q{args.jpeg_quality} | out={args.out}")

    cameras = tuple(args.cameras)
    cur_ep: int | None = None
    n_exported = 0
    buf: dict = {}

    def _new_buf(task: str) -> dict:
        return {
            "qpos": [], "effort": [], "action": [],
            "images": {cam: [] for cam in cameras},
            "task": task, "fps": float(ds.fps),
        }

    for i in range(ds.num_frames):
        s = ds[i]
        ep = int(s["episode_index"])
        if cur_ep is None:
            cur_ep = ep
            buf = _new_buf(str(s["task"]))
        elif ep != cur_ep:
            _flush_episode(args.out, cur_ep, buf, cameras)
            n_exported += 1
            if args.max_episodes is not None and n_exported >= args.max_episodes:
                print(f"Reached --max-episodes={args.max_episodes}; stopping early.")
                return
            cur_ep = ep
            buf = _new_buf(str(s["task"]))

        state = np.asarray(s["observation.state"], dtype=np.float32)
        buf["qpos"].append(state[POS_IDX])
        buf["effort"].append(state[EFF_IDX])
        buf["action"].append(np.asarray(s["action"], dtype=np.float32))
        for cam in cameras:
            hwc = _chw_float_to_hwc_uint8(s[f"observation.images.{cam}"])
            buf["images"][cam].append(_encode_jpeg(hwc, args.jpeg_quality))

        if (i + 1) % 1000 == 0:
            print(f"  ...processed {i + 1}/{ds.num_frames} frames")

    if cur_ep is not None and buf["action"]:
        _flush_episode(args.out, cur_ep, buf, cameras)
        n_exported += 1

    print(f"Done. Exported {n_exported} episodes to {args.out}")


if __name__ == "__main__":
    main()
