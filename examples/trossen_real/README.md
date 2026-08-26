# TA-VLA pi0 — real Trossen deploy client

Deploy the trained torque-aware **pi0** (`EXPERT_HIS_C_FUT`, config
`pi0_trossen_transfer_effort_sota`) on the real Trossen bimanual WidowX AI arms.

The model is an openpi **JAX/orbax** checkpoint, so it is served by openpi's own websocket
policy server. A thin robot client reuses `lerobot_robot_trossen` purely for hardware I/O and
maintains the 10-frame **effort-history buffer** the model expects. The two run as separate
processes (and separate venvs) on the machine with the GPU + arms, talking over
`ws://localhost:8000`.

```
Process A  (~/EVAN-TA-VLA/.venv, JAX)        Process B  (~/lerobot_trossen/.venv, py3.12 + openpi-client)
  scripts/serve_policy.py  <───ws:8000───>   examples/trossen_real/main.py
  loads checkpoints/.../29999                 bi_widowxai_follower_robot (arms + 3 RealSense cams)
```

## Observation / action contract (server-side transforms do the rest)
The client sends observations already in `TavlaInputs` form (the server does **not** repack):

| field    | shape / type      | content                                                            |
|----------|-------------------|--------------------------------------------------------------------|
| `state`  | `float32[14]`     | `[L joint_0..5, L carriage, R joint_0..5, R carriage]` (rad; grip m) |
| `effort` | `float32[10,14]`  | 10 history frames `[L ext_eff(7), R ext_eff(7)]` (Nm/N), oldest→newest at offsets `(-36,-32,…,0)` |
| `images` | 3× HWC uint8 RGB  | `cam_high`, `cam_left_wrist`, `cam_right_wrist` (server resizes→224) |
| `prompt` | str               | `"Grab and hand over the Rubik's cube to the other arm"`           |

The server returns `{"actions": np[50,14]}` already **absolute** (it un-deltas with the `state`
we send and strips the predicted future effort). `ActionChunkBroker` hands the client one
14-vector per tick, mapped back onto the follower's `.pos` action keys.

Because the same follower (`include_external_effort=True`) recorded the training data, joint
order and gripper-in-meters match by construction.

### Why `action_horizon = 25` (not 50)
The model was **trained** with `action_horizon = 50` — it always predicts a 50-step action
chunk (and 50 future-effort frames). At **deploy** time that is a separate knob: the client's
`ActionChunkBroker(action_horizon=25)` executes only the first 25 steps of each predicted chunk,
then re-queries the server with a fresh observation. This is intentional and safe:

- Each re-inference refreshes the **effort history** and **camera images**, so the torque signal
  the model conditions on never goes stale mid-chunk (important for a contact-rich handover).
- The returned actions are **absolute** and un-delta'd against the `state` sent *at inference
  time*, so consuming only a prefix of the chunk is correct — there is no drift from skipping the
  tail.
- Re-inferring every 25 ticks (~0.83 s at 30 Hz) trades a bit more compute for tighter reactivity.

Raising it toward 50 executes more open-loop steps per inference (less compute, less reactive);
lowering it re-infers more often. 25 is the recommended default; tune only if needed.

### Start pose (in-distribution)
On connect, both arms move to the data-collection start pose
(`STAGED_POSITIONS = [0, 0, 0, 0, 0, 0, 0]` in `env.py` — all-zeros, matching the pose the
training dataset was recorded from), so each episode begins inside the training distribution.

## Prerequisites (one-time)
Install the client deps into the lerobot venv (adds `openpi-client`, `tyro`, `websockets`,
`msgpack`, `dm-tree`):

```bash
uv pip install --python ~/lerobot_trossen/.venv/bin/python -e ~/EVAN-TA-VLA/packages/openpi-client tyro
```

Hardware config is read from the shell env vars (already set in `~/.bashrc`), with the known
values as fallbacks:
`FOLLOWER_LEFT_IP_ADDR`, `FOLLOWER_RIGHT_IP_ADDR`, `CAM_HIGH_SN`, `CAM_LEFT_WRIST_SN`,
`CAM_RIGHT_WRIST_SN`. Override any of them with the matching `--*` CLI flag.

## Run

### Process A — model server (`~/EVAN-TA-VLA/.venv`, JAX + GPU)
```bash
cd ~/EVAN-TA-VLA && .venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_transfer_effort_sota \
  --policy.dir checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999
```

> **RTX 5090 (Blackwell) note:** serving locally requires a JAX/CUDA stack that targets sm_120.
> The pinned `jax==0.5.0` (CUDA 12.6) fails with `ptxas too old` / `bf16→f16` errors. Fix (applied
> Jul 20 2026): upgrade the CUDA-12 libs to 12.9 and bump `jax`/`jaxlib` to `0.5.3` — see §10 of
> `RUNPOD_SETUP_AND_TRAINING.md`. Note `uv sync` reverts it.

> **Checkpoint present & verified (Jul 18 2026):** the trained SOTA checkpoint was rsync'd back
> from the pod to `checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999` — **5.8 GB**, intact
> orbax OCDBT (bulk weights in `params/ocdbt.process_0/d/`: ~2641 + 1804 + 1242 + 193 MB), valid
> `_METADATA`/`_sharding`, and embedded `assets/norm_stats.json` with keys `state(32)`,
> `actions(32)`, `effort(14)`. `serve_policy.py` loads norm stats from the checkpoint's own
> `assets/`, so no external norm-stats file is needed to serve.

### Process B — robot client (`~/lerobot_trossen/.venv`)
> tyro uses **hyphenated** flags (`--action-horizon`, `--dry-run`, `--max-episode-steps`, …).

```bash
cd ~/EVAN-TA-VLA && ~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 --dry-run   # drop --dry-run to move arms
```

Note: connecting drives both arms to their staged start pose (all-zeros) and opens the cameras —
even in `--dry-run`. `--dry-run` only suppresses the *policy* actions (arms won't follow the model).

## Evaluation walkthrough (run in this order)

### Pre-flight readiness (software) — verified Jul 20 2026
| Check | Command | Expected |
|---|---|---|
| JAX sees the GPU | `.venv/bin/python -c "import jax; print(jax.devices())"` | `[CudaDevice(id=0)]` (RTX 5090, ~22 GB free) |
| No competing GPU load | `nvidia-smi` | enough free VRAM for the 5.8 GB checkpoint |
| Checkpoint present | `du -sh checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999` | `5.8G` |
| Client stack imports | `~/lerobot_trossen/.venv/bin/python -c "import openpi_client, tyro, trossen_arm, pyrealsense2, lerobot_robot_trossen; import examples.trossen_real.main"` | no error |
| Hardware env vars set | `echo $FOLLOWER_LEFT_IP_ADDR $FOLLOWER_RIGHT_IP_ADDR $CAM_HIGH_SN $CAM_LEFT_WRIST_SN $CAM_RIGHT_WRIST_SN` | IPs + 3 serials |

> Run the JAX check **outside** any sandbox — a sandbox that blocks device access falsely reports
> `CpuDevice`, and serving on CPU would be unusably slow.

### Pre-flight (physical / safety) — before starting the client
1. Power on both follower arms; confirm reachable: `ping 192.168.1.5`, `ping 192.168.1.4`.
2. Plug in the 3 RealSense cameras (high, left wrist, right wrist).
3. **Clear the workspace and keep a hand on the E-stop** — the client drives both arms to the
   staged start pose (all-zeros) **on connect, even in `--dry-run`**.
4. Place the Rubik's cube at the same start location used during data collection.

### 1. Server smoke
Start Process A (server), then Process B with `--dry-run` — the client should log
`Server metadata: {...}` on connect.

### 2. Dry run (arms stage to start pose, but do NOT follow the policy)
```bash
cd ~/EVAN-TA-VLA && ~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 --max-episode-steps 150 --dry-run
```
Confirm: both arms move to the start pose (all-zeros), cameras open, and repeated
`[dry-run] action (not sent): [ …14 numbers… ]` look sane — no `nan`, arm joints ~±3 rad, gripper
dims (indices **6** and **13**) in a sane meters range. Arms stay still. `Ctrl-C` to stop.

### 3. Guarded live (arms follow the policy)
Drop `--dry-run`; start short and slow with the E-stop ready:
```bash
cd ~/EVAN-TA-VLA && ~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 \
  --max-episode-steps 150 --max-relative-target 1.0 --action-ema-alpha 0.5
```
- `--max-episode-steps 150` ≈ 5 s at 30 Hz for a first bounded attempt — raise toward `1000`
  (~33 s) once it looks safe.
- `--max-relative-target 1.0` keeps the follower's per-step joint clamp active (anti-lurch).
- `--action-ema-alpha 0.5` smooths commanded targets; use `1.0` to disable once trusted.

### 4. Stopping
`Ctrl-C` the **client first** (arms park at staged → sleep pose), then `Ctrl-C` the server.

### 5. Ablation compare (later)
Re-serve with `pi0_trossen_transfer_effort_expert` and run the same client to quantify the
torque-awareness lift at the handover contact moment.

## Key flags (`examples/trossen_real/main.py`)
| flag | default | purpose |
|---|---|---|
| `--host` / `--port` | `localhost` / `8000` | websocket policy server |
| `--action-horizon` | `25` | steps executed per inference chunk before re-querying (model predicts 50 — see "Why `action_horizon = 25`" above) |
| `--max-hz` | `30.0` | control rate — must match the 30 fps training rate |
| `--max-episode-steps` | `1000` | episode length bound |
| `--max-relative-target` | `1.0` | per-step joint clamp enforced by the follower (rad) |
| `--action-ema-alpha` | `1.0` | absolute-space action smoothing (1.0 = off) |
| `--dry-run` | off | assemble + query, never command the arms |
| `--prompt` | training string | must byte-match the training `default_prompt` — **see warning below** |

## ⚠️ `--prompt` defaults to the Rubik's-cube task

`env.py` hardcodes `DEFAULT_PROMPT = "Grab and hand over the Rubik's cube to the other arm"`, and
the client sends a `prompt` key on **every** observation. Server-side, `InjectDefaultPrompt` only
fills that key when it is **absent**, so the client always wins and `serve_policy.py
--default-prompt` is silently ignored.

Serving any policy other than the cube ones therefore **requires** passing `--prompt` with that
config's exact `default_prompt`. Getting it wrong produces no error — just a model acting on the
wrong instruction. Current values:

| Config | `--prompt` |
|---|---|
| `pi0_trossen_transfer_effort_sota` / `_expert` / `pi0_trossen_transfer_base` | `Grab and hand over the Rubik's cube to the other arm` (the default) |
| `pi0_trossen_charger_plugin_effort_sota` | `Unplug the charging cube from the power strip, plug it into the adjacent outlet, then turn the power strip switch on` |
