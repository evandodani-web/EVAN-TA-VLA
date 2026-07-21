# Base π0 Ablation (no TA-VLA) — Runbook

Train a **plain base π0** (LoRA) on the *exact same* Trossen handover dataset used by the
torque-aware configs, but with **none** of the TA-VLA effort machinery. This is the "π0 only"
ablation to quantify the lift from the torque additions in
`pi0_trossen_transfer_effort_sota` / `pi0_trossen_transfer_effort_expert`.

Companion to `RUNPOD_SETUP_AND_TRAINING.md` (SSH/setup/transfer) and
`TROSSEN_TAVLA_FINDINGS_AND_PLAN.md` (the *why*). This file is just the base-π0 delta.

Verified against the same **H100 80GB** pod, dataset already at
`/workspace/hf/lerobot/trossen_bimanual_transfer_cube_tavla`.

---

## 0. What makes this "base π0"

The new config `pi0_trossen_transfer_base` is identical to the two effort configs in every
hyperparameter — LoRA (`gemma_2b_lora` + `gemma_300m_lora`), `freeze_filter`, `ema_decay=None`,
`num_train_steps=30_000`, `pi0_base` weight loader, prompt, dataset — **except**:

| | effort configs | `pi0_trossen_transfer_base` |
|---|---|---|
| `effort_type` | `EXPERT_HIS_C_FUT` / `EXPERT` | `EffortType.NO` (the `Pi0Config` default) |
| `effort_history` | `(-36…0)` / `(0,)` | `()` (empty — effort never loaded) |
| `action_in/out_proj` width | `action_dim + effort_dim = 46` | `action_dim = 32` (standard π0) |
| `effort_proj` | present | **not built** |

With `effort_type=NO` the model is architecturally byte-for-byte base π0; with an empty
`effort_history` the data loader never reads/normalizes an `effort` column at all.

**No dataset rsync is needed** — the dataset on the pod is config-independent and is reused as-is.

---

## 1. Code changes (already applied in this repo)

### 1a. New config — `src/openpi/training/config.py`
Added right after the `pi0_trossen_transfer_effort_expert` block:

```python
TrainConfig(
    name="pi0_trossen_transfer_base",
    model=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotTavlaDataConfig(
        repo_id="trossen_bimanual_transfer_cube_tavla",
        default_prompt="Grab and hand over the Rubik's cube to the other arm",
        base_config=DataConfig(local_files_only=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
    freeze_filter=pi0.Pi0Config(
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
    ).get_freeze_filter(),
    ema_decay=None,
),
```
`effort_history` defaults to `()` in `LeRobotTavlaDataConfig`, so its `__post_init__` omits the
`effort` repack key and `prompt_from_task` is `False` (a default prompt is set).

### 1b. Data-loader guard — `src/openpi/training/data_loader.py`
`create_dataset` used to *unconditionally* set `delta_timestamps["observation.effort"]`. With an
empty `effort_history` that becomes `[]`, and lerobot 0.1.0 then calls `torch.stack([])` → crash.
The effort lines are now guarded (backward compatible — every effort config has a non-empty
`effort_history`, so their behavior is unchanged):

```python
if data_config.effort_history:
    delta_timestamps["observation.effort"] = [t / dataset_meta.fps for t in data_config.effort_history]
    if model_config.effort_type in (EffortType.EXPERT_FUT, EffortType.EXPERT_HIS_C_FUT, EffortType.EXPERT_HIS_C_L_FUT):
        delta_timestamps["observation.effort"] += [(t + 1) / dataset_meta.fps for t in range(model_config.action_horizon)]
```

---

## 2. Compute norm stats (ON the pod)

Norm stats are keyed by config name under `assets/<name>/`, so the base config needs its own.
Because `effort_history` is empty, `compute_norm_stats.py` produces **only** `state` and `actions`
(no `effort` key):

```bash
cd /workspace/EVAN-TA-VLA
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot
.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_trossen_transfer_base
```
Output: `assets/pi0_trossen_transfer_base/trossen_bimanual_transfer_cube_tavla/norm_stats.json`
with keys `state (32)`, `actions (32)` and **no** `effort`. (This pass decodes every frame, so it
can take a while on the pod — let it finish before training.)

---

## 3. Pre-train smoke check (ON the pod)

Confirm the config + norm stats resolve and the loader returns **no** effort:

```bash
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot
.venv/bin/python -c "
import dataclasses, numpy as np
import openpi.training.config as C
from openpi.training import data_loader as DL
cfg = dataclasses.replace(C.get_config('pi0_trossen_transfer_base'), batch_size=2)
loader = DL.create_data_loader(cfg, shuffle=True, num_batches=1, num_workers=0)
obs, actions = next(iter(loader))
print('effort is None:', obs.effort is None)               # True
print('state:', tuple(np.asarray(obs.state).shape), 'actions:', tuple(np.asarray(actions).shape))
"
```
Expect `effort is None: True`, `state: (2, 32)`, `actions: (2, 50, 32)`.

---

## 4. Train (ON the pod, inside tmux)

The first run downloads `pi0_base` weights from the public `s3://openpi-assets` bucket, so expect
a startup pause before step 0.

```bash
tmux new -s train_base                     # detach: Ctrl-b d | reattach: tmux attach -t train_base
cd /workspace/EVAN-TA-VLA
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot

.venv/bin/python scripts/train.py pi0_trossen_transfer_base \
  --exp-name run_001 \
  --wandb-enabled                          # omit / --no-wandb-enabled to disable W&B
```
- Checkpoints: `checkpoints/pi0_trossen_transfer_base/run_001/`.
- `num_train_steps=30_000`; saved every 1,000 steps, kept at multiples of 5,000.
- Resume an interrupted run: add `--resume` (mutually exclusive with `--overwrite`).
- Sanity-check the first ~50 steps (loss prints and trends down) before detaching.

---

## 5. Evaluate the trained base π0 on the real Trossen arms

Deployment runs on the **local machine** (it has the GPU **and** the arms), *not* the pod. Two
processes in two terminals talk over `ws://localhost:8000`: Terminal 1 = JAX model server,
Terminal 2 = robot client. This mirrors §8–§10 of `RUNPOD_SETUP_AND_TRAINING.md`, updated for the
base config. Run everything from `~/EVAN-TA-VLA`.

> The base model was trained with `effort_type=NO`, so the `effort` field the client still sends is
> **harmlessly ignored** server-side (the embedded norm stats have only `state`/`actions`; the
> server drops `effort` and `preprocess_observation` sets it to `None`). No client changes needed —
> the same `examples/trossen_real` client works unchanged for all three configs.

### 5a. Pull the checkpoint from the pod → local deploy machine (run from LOCAL)
```bash
# ~5-6 GB orbax checkpoint; adjust host/port to your pod (see RUNPOD_SETUP §0).
rsync -rltvP --mkpath --no-owner --no-group --no-perms -e "ssh -p 22145 -i ~/.ssh/id_ed25519" \
  root@63.141.33.87:/workspace/EVAN-TA-VLA/checkpoints/pi0_trossen_transfer_base/run_001/29999/ \
  ~/EVAN-TA-VLA/checkpoints/pi0_trossen_transfer_base/run_001/29999/
```

### 5b. Mirror the code edits to the local deploy machine (one-time)
The `config.py` + `data_loader.py` edits from §1 must exist in the **local** repo too (the serving
venv reads the same config). rsync the repo (excluding `.venv`/caches) or re-apply the two edits.

### 5c. One-time client install (skip if already done for the effort configs)
```bash
uv pip install --python ~/lerobot_trossen/.venv/bin/python -e ~/EVAN-TA-VLA/packages/openpi-client tyro
```

### 5d. (RTX 5090 / Blackwell only) re-apply the JAX/CUDA bump
If serving on the local RTX 5090, the pinned `jax==0.5.0` stack can't codegen sm_120. Re-run the
two `uv pip install` commands from §10 of `RUNPOD_SETUP_AND_TRAINING.md` (needed again after any
`uv sync`). H100/other GPUs can skip this.

### 5e. Pre-flight readiness (software) — run OUTSIDE any sandbox so the GPU is visible
| Check | Command | Expected |
|---|---|---|
| JAX sees the GPU | `.venv/bin/python -c "import jax; print(jax.devices())"` | `[CudaDevice(id=0)]` |
| No competing GPU load | `nvidia-smi` | enough free VRAM for the ~5.8 GB checkpoint |
| Checkpoint present | `du -sh checkpoints/pi0_trossen_transfer_base/run_001/29999` | ~5.8G |
| Client stack imports | `~/lerobot_trossen/.venv/bin/python -c "import openpi_client, tyro, trossen_arm, pyrealsense2, lerobot_robot_trossen; import examples.trossen_real.main"` | no error |
| Hardware env vars set | `echo $FOLLOWER_LEFT_IP_ADDR $FOLLOWER_RIGHT_IP_ADDR $CAM_HIGH_SN $CAM_LEFT_WRIST_SN $CAM_RIGHT_WRIST_SN` | the 2 IPs + 3 serials |

### 5f. Pre-flight (physical / safety) — before starting the client
1. Power on both follower arms; confirm reachable: `ping 192.168.1.5` and `ping 192.168.1.4`.
2. Plug in the 3 RealSense cameras (high, left wrist, right wrist).
3. **Clear the workspace and keep a hand on the E-stop** — the client drives both arms to the
   staged start pose (all-zeros) **on connect, even in `--dry-run`**.
4. Place the Rubik's cube at the same start location used during data collection.

### 5g. Terminal 1 — start the model server (JAX venv)
```bash
cd ~/EVAN-TA-VLA
.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_transfer_base \
  --policy.dir checkpoints/pi0_trossen_transfer_base/run_001/29999
```
Wait for `Creating server (host: …)`. Loading is fully local (weights + PaliGemma tokenizer are
already cached — no internet needed). Serves on `0.0.0.0:8000`.

### 5h. Terminal 2 — dry run (arms stage to start pose, do NOT follow the policy)
```bash
cd ~/EVAN-TA-VLA
~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 \
  --max-episode-steps 150 --dry-run
```
Success signals: client logs `Server metadata: {...}` (websocket connected); both arms move to the
all-zeros start pose; cameras open; repeated `[dry-run] action (not sent): [ …14 numbers… ]` —
eyeball them (no `nan`, arm joints ~±3 rad, gripper dims at indices **6** and **13** in a sane
meters range). Arms stay still. `Ctrl-C` to stop.

### 5i. Terminal 2 — guarded live run (arms follow the policy)
Start short and slow; E-stop ready:
```bash
cd ~/EVAN-TA-VLA
~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 \
  --max-episode-steps 150 \
  --max-relative-target 1.0 \
  --action-ema-alpha 0.5
```
- `--max-episode-steps 150` ≈ 5 s at 30 Hz for a first bounded attempt — raise toward `1000`
  (~33 s) once it looks safe.
- `--max-relative-target 1.0` keeps the follower's per-step joint clamp active (anti-lurch).
- `--action-ema-alpha 0.5` smooths commanded joint targets; use `1.0` to disable once trusted.
- `--action-horizon 25` is a deploy knob (the model always predicts a 50-step chunk; the broker
  executes 25 then re-infers). Same value used for the effort configs, for a fair comparison.

### 5j. Stopping
`Ctrl-C` the **client first** (arms park at staged → sleep pose), then `Ctrl-C` the server.

---

## 6. Ablation comparison

Three directly comparable runs on the identical dataset (same LoRA / steps / prompt), each served
with its own config name + checkpoint and driven by the **same** `examples/trossen_real` client:

| Config | `effort_type` | Torque additions | Serve `--policy.config` / `--policy.dir` |
|---|---|---|---|
| `pi0_trossen_transfer_base` | `NO` | none (this run) | `pi0_trossen_transfer_base` / `checkpoints/pi0_trossen_transfer_base/run_001/29999` |
| `pi0_trossen_transfer_effort_expert` | `EXPERT` | current-frame torque token (obs) | `pi0_trossen_transfer_effort_expert` / `checkpoints/pi0_trossen_transfer_effort_expert/run_001/29999` |
| `pi0_trossen_transfer_effort_sota` | `EXPERT_HIS_C_FUT` | history token + future-torque objective (obs+obj) | `pi0_trossen_transfer_effort_sota` / `checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999` |

To compare: for each config, restart Terminal 1 with that config's `--policy.config` +
`--policy.dir`, then run the identical client command (§5i) with the same seed pose / cube
placement / `--action-horizon` / `--max-episode-steps`. Score handover success and the contact
moment (grasp stability, over-squeeze, hand-off timing) across the three to quantify the TA-VLA
lift over base π0.
