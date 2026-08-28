# RunPod Setup & Training Runbook — TA-VLA (Trossen handover)

Step-by-step commands to go from a local machine + a fresh RunPod H100 pod to a running
training job for the torque-aware π0 configs. Companion to
`TROSSEN_TAVLA_FINDINGS_AND_PLAN.md` (which covers the *why*); this file is the *how*.

Verified end-to-end on **Jul 13 2026**: RunPod **H100 80GB HBM3**, `/workspace` network volume.

---

## 0. Reference values (edit for your pod)

| Thing | Value used |
|---|---|
| Pod SSH host | `63.141.33.87` |
| Pod SSH port | `22145` |
| Pod user | `root` |
| Local SSH key | `~/.ssh/id_ed25519` (pubkey `~/.ssh/id_ed25519.pub`) |
| Local dataset | `~/.cache/huggingface/lerobot/trossen_bimanual_transfer_cube_tavla` |
| Local repo | `~/EVAN-TA-VLA` |
| Pod dataset dir | `/workspace/hf/lerobot/trossen_bimanual_transfer_cube_tavla` |
| Pod repo dir | `/workspace/EVAN-TA-VLA` |
| Pod `LEROBOT_HOME` (+ `HF_LEROBOT_HOME`) | `/workspace/hf/lerobot` |

RunPod also exposes extra TCP ports (e.g. `:22146`, `:2022`); only the SSH port `22145`
matters for Cursor Remote-SSH and rsync. `:22146` is RunPod's HTTP proxy for internal port
`:2022` (Eternal Terminal) — but RunPod's proxy is HTTP-only, so raw TCP (ET) is blocked
externally. Use an SSH tunnel instead (see §1e).

---

## 1. SSH access

### 1a. Add a host alias in `~/.ssh/config` (local machine)
```
Host runpod-tavla
    HostName 63.141.33.87
    Port 22145
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 4
```
After this, `ssh runpod-tavla` works as shorthand for the full command.

### 1b. Authorize this machine's key on the pod
RunPod injects **account-level** SSH keys into the pod at startup. If this machine's key
isn't registered, you'll get `Permission denied (publickey)`. Two ways to add it:

- **Option A (persists across pods):** RunPod dashboard → Settings → SSH Public Keys →
  paste the contents of `~/.ssh/id_ed25519.pub` on its own line → **restart the pod**.
- **Option B (running pod, immediate):** open the RunPod **web terminal** and run
  (paste your actual pubkey line — multiple keys are fine, one per line, `>>` appends):
```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo 'ssh-ed25519 AAAA...your-pubkey... comment' >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```
Multiple machines (laptop + workstation) can each have their key authorized simultaneously.

### 1c. Verify the connection
```bash
ssh runpod-tavla "whoami; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /workspace"
```
Expect: `root`, an H100 line, and the `/workspace` mount. To debug a rejected key
(without exposing it), match fingerprints — local `ssh-keygen -lf ~/.ssh/id_ed25519.pub`
should appear in the pod's `ssh-keygen -lf ~/.ssh/authorized_keys`.

### 1d. Connect Cursor (optional IDE access)
`Ctrl+Shift+P` → **Remote-SSH: Connect to Host…** → `runpod-tavla` → platform **Linux** →
**File → Open Folder** → `/workspace/EVAN-TA-VLA`.

### 1e. Eternal Terminal (optional — reconnectable shell)
ET lets you reconnect to a running shell after a network drop without re-running `tmux attach`.
RunPod's port 22146 is an HTTP proxy (not raw TCP), so ET must tunnel through SSH.

**On the pod** (once per pod lifetime, after SSH-ing in):
```bash
etserver --daemon   # listens on port 2022 (default); do NOT use --port 22146
```

**From your Mac** (two steps each session):
```bash
# 1. Open the SSH tunnel in the background
ssh -fNL 2022:localhost:2022 root@63.141.33.87 -p 22145 -i ~/.ssh/id_ed25519

# 2. Connect ET through it
et root@localhost:2022
```

> Note: `tmux` (§5) already handles disconnects for training jobs and is simpler.
> ET is only worth the tunnel overhead if you want seamless shell reconnection.

---

## 2. Transfer data to the pod (run from the LOCAL machine)

`/workspace` is a **MooseFS network volume** that does not allow `chown`, so plain `rsync -a`
prints many `chown ... Operation not permitted` warnings and exits with `code 23`. These are
**harmless** (file contents transfer fine). Avoid the noise by dropping owner/group/perms
preservation: use `-rltvP --no-owner --no-group --no-perms`. `--mkpath` creates missing
parent dirs on the receiver.

### 2a. Dataset (~1 GB AV1 video; `-z` omitted — already compressed)
```bash
rsync -rltvP --mkpath --no-owner --no-group --no-perms -e "ssh -p 22145 -i ~/.ssh/id_ed25519" \
  ~/.cache/huggingface/lerobot/trossen_bimanual_transfer_cube_tavla/ \
  root@63.141.33.87:/workspace/hf/lerobot/trossen_bimanual_transfer_cube_tavla/
```

### 2b. Repo — includes `assets/` norm stats (excludes venv/caches/git)
```bash
rsync -rltvP --mkpath --no-owner --no-group --no-perms \
  --exclude='.venv' --exclude='__pycache__' --exclude='.git' -e "ssh -p 22145 -i ~/.ssh/id_ed25519" \
  ~/EVAN-TA-VLA/ \
  root@63.141.33.87:/workspace/EVAN-TA-VLA/
```
> The `assets/pi0_trossen_transfer_effort_sota/` and `assets/pi0_trossen_transfer_effort_expert/`
> norm-stat dirs were computed locally and **must** ride along (they're keyed by config name).
> Alternatively, recompute them on the pod (see §4b).

### 2c. Verify (from local, over SSH)
```bash
ssh runpod-tavla "ls /workspace/hf/lerobot/trossen_bimanual_transfer_cube_tavla && echo '--- assets ---' && ls /workspace/EVAN-TA-VLA/assets"
```
Expect `data  meta  videos` and the two `pi0_trossen_transfer_effort_*` dirs.

---

## 3. Environment setup (run ON the pod)
```bash
cd /workspace/EVAN-TA-VLA
# IMPORTANT: openpi's pinned lerobot 0.1.0 reads LEROBOT_HOME (NOT HF_LEROBOT_HOME).
# Set both so the data loader (0.1.0) and any 0.5.2 tooling both resolve the dataset.
echo 'export HF_LEROBOT_HOME=/workspace/hf/lerobot' >> ~/.bashrc
echo 'export LEROBOT_HOME=/workspace/hf/lerobot'    >> ~/.bashrc
source ~/.bashrc
uv sync                                   # builds .venv (lerobot 0.1.0 + JAX)
```

Verify JAX sees the GPU (must print a `CudaDevice`, not `CpuDevice`):
```bash
.venv/bin/python -c "import jax; print(jax.devices())"
# -> [CudaDevice(id=0)]
```

---

## 4. Pre-training checks (ON the pod)

### 4a. Config + norm stats resolve
```bash
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot
.venv/bin/python -c "
import openpi.training.config as C
cfg = C.get_config('pi0_trossen_transfer_effort_sota')
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
print('effort_dim_in =', cfg.model.effort_dim_in)              # 140
print('norm_stats keys =', None if dc.norm_stats is None else list(dc.norm_stats))
"
```
Expect `effort_dim_in = 140` and `norm_stats keys = ['state', 'actions', 'effort']`.
If `norm_stats keys` is `None`, the assets didn't transfer — recompute (§4b).

> Note: this check reads norm stats from `assets/` only — it does **not** load the dataset,
> so it will pass even if `LEROBOT_HOME` is wrong. Use §4c to actually exercise the data loader.

### 4b. (Only if norm stats are missing) recompute on the pod
```bash
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot
.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_trossen_transfer_effort_sota
.venv/bin/python scripts/compute_norm_stats.py --config-name=pi0_trossen_transfer_effort_expert
```

### 4c. Data-loader smoke test (catches a wrong `LEROBOT_HOME`)
```bash
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot
.venv/bin/python -c "
import dataclasses, numpy as np
import openpi.training.config as C
from openpi.training import data_loader as DL
cfg = dataclasses.replace(C.get_config('pi0_trossen_transfer_effort_sota'), batch_size=2)
loader = DL.create_data_loader(cfg, shuffle=True, num_batches=1, num_workers=0)
obs, actions = next(iter(loader))
print('effort:', tuple(np.asarray(obs.effort).shape))   # (2, 60, 14) = 10 history + 50 future
print('state:', tuple(np.asarray(obs.state).shape), 'actions:', tuple(np.asarray(actions).shape))
"
```
Expect `effort: (2, 60, 14)`, `state: (2, 32)`, `actions: (2, 50, 32)`. A `FileNotFoundError`
for `meta/info.json` under `~/.cache/huggingface/lerobot/...` means `LEROBOT_HOME` isn't set.

---

## 5. Train (ON the pod, inside tmux)

The first run downloads `pi0_base` weights from the public `s3://openpi-assets` bucket
(needs outbound internet), so expect a startup pause before step 0.

```bash
tmux new -s train                         # detach: Ctrl-b d | reattach: tmux attach -t train
cd /workspace/EVAN-TA-VLA
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot

# SOTA variant — paper "+obs+obj" (EXPERT_HIS_C_FUT)
# NOTE: train.py uses tyro's overridable_config_cli — the config name is a POSITIONAL
# subcommand (no --config-name=), and --wandb-enabled is a bare boolean flag (no =True).
.venv/bin/python scripts/train.py pi0_trossen_transfer_effort_sota \
  --exp-name run_001 \
  --wandb-enabled
# For W&B: do NOT use `wandb login` (the interactive prompt rejects new-format
# `wandb_v1_...` keys). Instead set the key as an env var:
#   echo 'export WANDB_API_KEY=<your wandb_v1_ key>' >> ~/.bashrc && source ~/.bashrc
# Omit --wandb-enabled (or pass --no-wandb-enabled) to disable W&B.
```

Ablation baseline (pure-obs DePost, `EXPERT`) — same command, different config:
```bash
.venv/bin/python scripts/train.py pi0_trossen_transfer_effort_expert \
  --exp-name run_001 \
  --wandb-enabled
```

Notes:
- `--exp-name` is **required**; names the checkpoint dir
  `checkpoints/<config_name>/<exp_name>/`.
- `num_train_steps=30_000`; checkpoints every 1,000 steps, kept at multiples of 5,000.
- Resume an interrupted run: add `--resume` (bare boolean flag; mutually exclusive with `--overwrite`).
- Sanity-check the first ~50 steps: loss should print and trend down before you detach.

---

## 6. The two configs (defined in `src/openpi/training/config.py`)

| Config name | `effort_type` | `effort_history` | `effort_dim_in` | Purpose |
|---|---|---|---|---|
| `pi0_trossen_transfer_effort_sota` | `EXPERT_HIS_C_FUT` | `(-36…0)` step 4, len 10 | 140 | paper SOTA "+obs+obj" |
| `pi0_trossen_transfer_effort_expert` | `EXPERT` | `(0,)` | 14 | pure-obs "DePost" ablation |

Both: `repo_id="trossen_bimanual_transfer_cube_tavla"`, `local_files_only=True`,
`default_prompt="Grab and hand over the Rubik's cube to the other arm"`, LoRA freeze filter,
`CheckpointWeightLoader(pi0_base)`, `action_horizon=50`, `num_train_steps=30_000`,
`ema_decay=None`.

---

## 7. Troubleshooting quick reference

| Symptom | Cause / fix |
|---|---|
| `Permission denied (publickey)` | Pod doesn't have this machine's pubkey — see §1b (Option B for a running pod). |
| rsync `mkdir ... No such file or directory` | Missing parent dir — add `--mkpath`. |
| rsync `chown ... Operation not permitted` + `code 23` | Harmless on the `/workspace` network FS — use `--no-owner --no-group --no-perms`. |
| `jax.devices()` shows `CpuDevice` | GPU not visible to JAX — check the pod actually has a GPU / correct JAX build in `.venv`. |
| `norm_stats keys = None` at train start | Norm-stat assets not present under `assets/<config>/` — transfer them or recompute (§4b). |
| `FileNotFoundError: .../meta/info.json` under `~/.cache/huggingface/lerobot` | `LEROBOT_HOME` not set. openpi's lerobot 0.1.0 reads `LEROBOT_HOME` (not `HF_LEROBOT_HOME`); `export LEROBOT_HOME=/workspace/hf/lerobot`. |
| `wandb: ERROR API key must be 40 characters long` | New-format `wandb_v1_...` key pasted into interactive `wandb login`. Use `export WANDB_API_KEY=...` instead. |
| Training dies on SSH disconnect | Run inside `tmux` (§5). |
| `ptxas too old` / `ptxas does not support CC 12.0` when serving locally | Local **RTX 5090 is Blackwell (sm_120)**; the pinned CUDA/JAX stack is too old. Run `scripts/fix_blackwell_cuda.sh` (§10). Recurs after every `uv sync`. |
| Segfault (exit 139) in any matmul, but `jax.devices()` looks fine | CUDA 12.9 libs on a driver that is too old — e.g. the Blackwell fix applied to the A100 pod (driver 550.90.12). `uv sync` restores a working stack; see §10a. |
| `LLVM ERROR: Unsupported conversion from bf16 to f16` when serving locally | Same Blackwell issue — jaxlib 0.5.0's XLA can't codegen sm_120. Bumping `jax`/`jaxlib` to `0.5.3` fixes it (§10). |
| Client: `Unrecognized options: --host … --port …` (tyro wants `--args.host`) | Newer tyro nests a function's dataclass param under its name. `examples/trossen_real/main.py` parses the dataclass directly (`main(tyro.cli(Args))`, like `serve_policy.py`) so flags are `--host`/`--port`/… as documented. Already fixed. |
| `compute_norm_stats.py` runs for hours | It sets `num_batches = num_frames` but reads only `batch[key][0]`, so it loads 8 samples per used frame and cycles the dataset 8×. Bound it with `--max-frames` (§11c). |
| Whole pod stops responding; even `echo` hangs in a new shell | Sustained multi-hundred-GB reads through the `/workspace` MooseFS FUSE mount inside the ~117 GB container cgroup. Bound the read volume and keep the dataset on **local** disk (§11d). It recovered on its own after ~1 h; a pod restart also clears it (`/workspace` is persistent). |
| Training ~5 s/it and `nvidia-smi` util oscillates 0%/100% | Data-starved, not compute-bound. `TrainConfig.num_workers` defaults to **2** and the dataset is on the network mount. Copy the dataset to local disk and pass `--num-workers 8` (§11d). |
| Loss flat for the first ~100 steps | Expected: `CosineDecaySchedule` warms up over **1,000 steps** from `peak_lr/1001` ≈ 2.5e-8. Judge the curve at step ≥1,000, not before. |
| Prompt silently truncated | `Pi0Config.max_token_len = 48` including BOS + `\n`; the tokenizer only logs a warning. Check a new task's prompt fits (§11b). |
| `pkill -f "scripts/train.py"` kills your own shell | The pattern matches the command line of the shell running `pkill`. Prefer `tmux kill-session -t train`, or a pattern that can't self-match. |

---

## 8. Next: deployment (Phase 3) — client implemented
The deploy client now exists: **`examples/trossen_real/`** (`env.py`, `main.py`, `README.md`).
After training, rsync the checkpoint back from the pod to this machine (which has the GPU **and**
the arms), then run the two-process deploy:

```bash
# From the LOCAL machine: pull the trained checkpoint back from the pod (~5.8 GB)
rsync -rltvP --mkpath --no-owner --no-group --no-perms -e "ssh -p 22145 -i ~/.ssh/id_ed25519" \
  root@63.141.33.87:/workspace/EVAN-TA-VLA/checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999/ \
  ~/EVAN-TA-VLA/checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999/
```

```bash
# One-time: install the client deps into the lerobot venv
uv pip install --python ~/lerobot_trossen/.venv/bin/python -e ~/EVAN-TA-VLA/packages/openpi-client tyro

# Process A — model server (JAX venv)
cd ~/EVAN-TA-VLA && .venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_transfer_effort_sota \
  --policy.dir checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999

# Process B — robot client (lerobot venv); drop --dry-run to move the arms
cd ~/EVAN-TA-VLA && ~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 --dry-run
```

The client maintains the 10-frame effort-history buffer at the training offsets, stages the arms
to the data-collection start pose (all-zeros), and enforces the follower `max_relative_target` clamp (plus
optional `--action-ema-alpha`). Full contract, `action_horizon` rationale, and the safety ladder
are in `examples/trossen_real/README.md` and Phase 3 of `TROSSEN_TAVLA_FINDINGS_AND_PLAN.md`.
Checkpoint verified present locally (5.8 GB, intact orbax OCDBT) on Jul 18 2026.

---

## 9. Evaluate the policy on the real Trossen arms — step by step

This is the deploy on **this machine** (it has the GPU **and** the arms). Two processes in two
terminals talk over `ws://localhost:8000`: Terminal 1 = JAX model server, Terminal 2 = robot
client. Run everything from `~/EVAN-TA-VLA`.

### 9a. Pre-flight readiness (software) — verified Jul 20 2026
| Check | Command | Expected |
|---|---|---|
| JAX sees the GPU | `.venv/bin/python -c "import jax; print(jax.devices())"` | `[CudaDevice(id=0)]` (RTX 5090, ~22 GB free) |
| No competing GPU load | `nvidia-smi` | enough free VRAM for the 5.8 GB checkpoint |
| Checkpoint present | `du -sh checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999` | `5.8G` |
| Client stack imports | `~/lerobot_trossen/.venv/bin/python -c "import openpi_client, tyro, trossen_arm, pyrealsense2, lerobot_robot_trossen; import examples.trossen_real.main"` | no error |
| Hardware env vars set | `echo $FOLLOWER_LEFT_IP_ADDR $FOLLOWER_RIGHT_IP_ADDR $CAM_HIGH_SN $CAM_LEFT_WRIST_SN $CAM_RIGHT_WRIST_SN` | `192.168.1.5 192.168.1.4 419122270126 412622272448 412622272396` |

> The JAX check must be run **outside** any sandbox — a sandbox that blocks device access will
> falsely report `CpuDevice`. Serving on CPU would be unusably slow.

### 9b. Pre-flight (physical / safety) — do these before starting the client
1. Power on both follower arms; confirm reachable: `ping 192.168.1.5` and `ping 192.168.1.4`.
2. Plug in the 3 RealSense cameras (high, left wrist, right wrist).
3. **Clear the workspace and keep a hand on the E-stop** — the client drives both arms to the
   staged start pose (all-zeros) **on connect, even in `--dry-run`**.
4. Place the Rubik's cube at the same start location used during data collection.
5. **Confirm every near-constant joint is staged at its training value.** Any dimension with a
   near-zero norm-stat std (e.g. the charger dataset's left gripper) normalizes by ~`1e-6`, so a
   small physical offset becomes a huge out-of-distribution input. See §11e for the check.

### 9c. Terminal 1 — start the model server (JAX venv)
```bash
cd ~/EVAN-TA-VLA
.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_transfer_effort_sota \
  --policy.dir checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999
```
Wait for `Creating server (host: …)`. Loading is fully local (weights + PaliGemma tokenizer are
already cached — no internet needed). Serves on `0.0.0.0:8000`.

### 9d. Terminal 2 — dry run (arms stage to start pose, but do NOT follow the policy)
```bash
cd ~/EVAN-TA-VLA
~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 \
  --max-episode-steps 150 --dry-run
```
Success signals:
- Client logs `Server metadata: {...}` (websocket connected).
- Both arms move to the start pose (all-zeros); cameras open.
- Repeated `[dry-run] action (not sent): [ …14 numbers… ]` — eyeball them: no `nan`, arm joints
  roughly in radians (~±3), gripper dims (indices **6** and **13**) in a sane meters range. Arms
  stay still (policy actions are suppressed). `Ctrl-C` to stop.

> **✅ Dry run verified (Jul 20 2026):** both arms connected (`192.168.1.5` / `192.168.1.4`), all 3
> RealSense cameras opened, 150 steps of inference ran and returned clean 14-dim actions (no NaNs),
> and the client disconnected cleanly (`Episode completed` → arms parked) — a strong sign the model
> loaded and decoded correctly. (That run used the earlier perch staging; the staged start pose has
> since been corrected to **all-zeros** to match the training dataset — see §10 note below / `env.py`.)

### 9e. Terminal 2 — guarded live run (arms follow the policy)
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

### 9f. Stopping
`Ctrl-C` the **client first** (arms park at staged → sleep pose), then `Ctrl-C` the server.

### 9g. Optional / later
- **Richer dry-run validation:** add a one-time first-step log of obs shapes (`state[14]`,
  `effort[10,14]`, 3 image shapes) + explicit NaN/range checks in `env.py` to harden the gate.
- **Ablation comparison:** re-serve with `pi0_trossen_transfer_effort_expert` (once that
  checkpoint exists) and run the same client to quantify the torque-awareness lift.

---

## 10. Serving locally on an RTX 5090 (Blackwell / sm_120) — JAX/CUDA fix ✅ applied Jul 20 2026

The training venv (`~/EVAN-TA-VLA/.venv`) was built for the **H100 pod (sm_90)** and pins
`jax[cuda12]==0.5.0`, whose bundled CUDA is 12.6. The **deploy machine's RTX 5090 is Blackwell
(compute capability 12.0 / sm_120)**, which needs **CUDA ≥ 12.8** *and* an XLA that can codegen
sm_120. Serving the checkpoint locally on 0.5.0 fails in two stages:

1. `ptxas too old` / `ptxas does not support CC 12.0` — the CUDA 12.6 `ptxas` can't target sm_120.
2. After bumping CUDA libs: `LLVM ERROR: Unsupported conversion from bf16 to f16` — jaxlib 0.5.0's
   XLA itself lacks Blackwell codegen.

**Fix (both steps required), run in the openpi venv:**
```bash
cd ~/EVAN-TA-VLA
# 1) Modernize the CUDA-12 userspace libs to 12.9 (ptxas/cuBLAS/cuDNN/…). The 0.5.x CUDA plugin
#    pins these as ">=", so this is allowed.
uv pip install --python .venv/bin/python -U \
  nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-cupti-cu12 nvidia-cublas-cu12 "nvidia-cudnn-cu12<10.0" \
  nvidia-cufft-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 \
  nvidia-curand-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12

# 2) Bump JAX to 0.5.3 (ships CUDA 12.8, has sm_120 codegen). Stays patch-level within 0.5.x so
#    openpi + flax==0.10.2 are unaffected.
uv pip install --python .venv/bin/python -U "jax[cuda12]==0.5.3"
```

**Verify (must run OUTSIDE any sandbox so the GPU is visible):**
```bash
# GPU compiles for sm_120
.venv/bin/python -c "import jax, jax.numpy as jnp; k=jax.random.key(0); x=jax.random.normal(k,(2048,2048)); print(jax.devices(), float((x@x).sum()))"
# Full checkpoint load + one inference -> actions (50, 14), no NaNs
.venv/bin/python -c "
import numpy as np, pathlib, openpi.training.config as C
from openpi.policies import policy_config as PC
p = PC.create_trained_policy(C.get_config('pi0_trossen_transfer_effort_sota'),
    pathlib.Path('checkpoints/pi0_trossen_transfer_effort_sota/run_001/29999'))
o = {'state': np.zeros(14,np.float32), 'effort': np.zeros((10,14),np.float32),
     'images': {c: np.zeros((480,640,3),np.uint8) for c in ('cam_high','cam_left_wrist','cam_right_wrist')},
     'prompt': \"Grab and hand over the Rubik's cube to the other arm\"}
a = np.asarray(p.infer(o)['actions']); print('actions', a.shape, 'NaN', bool(np.isnan(a).any()))
"
```

> **⚠️ `uv sync` reverts this.** Re-running `uv sync` restores the pinned `jax==0.5.0` + CUDA 12.6
> and re-breaks local serving. Rather than retyping the two commands, run
> **`scripts/fix_blackwell_cuda.sh`** after any `uv sync`; `--check` verifies without changing
> anything. It checks jax, `ptxas` *and* a real GPU matmul, and refuses to run on a machine whose
> driver is too old (see below).

### 10a. This fix is for the **Blackwell deploy box only** — it breaks the A100 pod

Tested Aug 28 2026: applying the fix on the **A100 pod (driver 550.90.12, max CUDA 12.4)** leaves
`import jax` and `jax.devices()` working — reporting `[CudaDevice(id=0)]` — while **any matmul
≥ 512×512 SIGSEGVs inside cuBLAS** (exit 139). A 4×4 matmul and `jax.random.normal` still succeed,
so a superficial check looks fine and training/serving would die later. `uv sync` restores a
working stack.

The CUDA 12.9 userspace libraries need a driver new enough to support them; the verified 5090 box
runs **driver 580.159.03**. The earlier claim that this bump is "H100-safe too" is therefore only
true where the driver is recent enough — it is a **driver** constraint, not an architecture one.
`fix_blackwell_cuda.sh` enforces this by refusing to apply when `nvidia-smi` reports a max CUDA
below 12.9 (override with `--force`).

**Why this is not pinned in `pyproject.toml`.** Bumping the pin there has no effect unless
`uv.lock` is regenerated, and re-resolving currently fails on a pre-existing, unrelated conflict:
`lerobot 0.1.0` requires `pyav`, which has no wheel for the Python 3.13 + macOS split that
`requires-python = ">=3.11"` forces uv to solve. (`uv lock` appears to succeed today only because
it validates the existing lock rather than re-resolving.) Fixing that means narrowing
`requires-python` or restricting resolution to Linux, then re-resolving all 254 packages — which
would change the environment both charger policies were trained in. The script is the contained
alternative.

Verified Jul 20 2026: RTX 5090, driver 580.159.03, `jax 0.5.3`, CUDA 12.9 libs → checkpoint
served on `0.0.0.0:8000`, inference returns `actions (50, 14)`.

**Recurred and re-applied Aug 26 2026.** The venv had reverted to `jax 0.5.0` + `ptxas 12.6.85`
(CUDA libs at 12.4), and loading the charger checkpoint died at
`nnx.eval_shape(self.create, jax.random.key(0))` with
`XlaRuntimeError: UNIMPLEMENTED: …/ptxas ptxas too old. Falling back to the driver to compile.`,
preceded by the warning `ptxas does not support CC 12.0`. Two things worth knowing:

- **Checking `jax.__version__` alone is not sufficient.** The CUDA libs and `jax` revert
  independently, so verify `ptxas --version` reports `release 12.9` as well:
  ```bash
  .venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas --version | tail -2
  ```
- **Step 2 also pulls newer `numpy`/`scipy`/`ml-dtypes`** (`numpy 2.0.2 → 2.4.6`,
  `scipy 1.14.1 → 1.17.1`, `ml-dtypes 0.5.0 → 0.6.0`) as transitive deps of `jax 0.5.3`. Serving is
  unaffected — re-verified on the charger checkpoint: `actions (50, 14)`, no NaNs, `absmax 0.17`.

---

## 11. Training a **new dataset** — deltas + performance notes

§1–§7 assume the Rubik's-cube dataset. This section is the delta for standing up a *different
task*, using the **charger plug-in** run as the worked example (Aug 25 2026, RunPod **A100 80GB
PCIe**). Read it before starting any new task — the two performance items (§11c, §11d) together
cut that run from an estimated 40 h to ~23 h.

### 11a. What actually changes per dataset

| Thing | Cube | Charger |
|---|---|---|
| v3.0 source dir | `trossen-bimanual-transfer-cube-external-effort-v2` | `trossen-bimanual-charger-plugin-external-effort` |
| npz intermediate (`--out`) | `~/tavla_intermediate` | `~/tavla_intermediate_charger` |
| v2.x `--repo-id` | `trossen_bimanual_transfer_cube_tavla` | `trossen_bimanual_charger_plugin_tavla` |
| Train config (SOTA) | `pi0_trossen_transfer_effort_sota` | `pi0_trossen_charger_plugin_effort_sota` |
| Train config (base ablation) | `pi0_trossen_transfer_base` | `pi0_trossen_charger_plugin_base` |
| Episodes / frames | 50 / 39,650 | 56 / 40,940 |

Both configs for a task share one dataset — the base ablation reuses it untouched, since an empty
`effort_history` means the effort column is never repacked or sampled. Only norm stats are
per-config (they are keyed by config name), and the base run's contain no `effort` key.

`scripts/tavla_data/*.py` need **no code edits** provided the source is the same bimanual rig —
28-dim `[L pos(7), L ext_eff(7), R pos(7), R ext_eff(7)]` state, 14-dim action, 4 cameras
(`cam_low` is dropped by `DEFAULT_CAMERAS`). Only the three CLI paths above change.

> `npz_to_lerobot_v2.py --repo-id` **defaults to the cube name**. Always pass it explicitly or
> you will silently overwrite the cube dataset. Likewise give `--out` a fresh directory so you
> don't mix `episode_*.npz` from two tasks.

The new `TrainConfig` is a copy of the cube SOTA block with `repo_id` and `default_prompt`
swapped — everything else (LoRA variants, `effort_type=EXPERT_HIS_C_FUT`, `effort_dim=14`,
`effort_history`, 30k steps, freeze filter) stays identical so tasks remain comparable.

### 11b. Check the prompt fits the token budget

`Pi0Config.max_token_len = 48` **including** BOS and the trailing `\n`, and the tokenizer only
logs a warning before truncating — a long task string can be silently cut. Verify before training:

```bash
.venv/bin/python -c "
from openpi.models import tokenizer as T
import openpi.training.config as C
cfg = C.get_config('pi0_trossen_charger_plugin_effort_sota')
tok = T.PaligemmaTokenizer(cfg.model.max_token_len)
print('tokens used:', int(tok.tokenize(cfg.data.default_prompt)[1].sum()), '/', cfg.model.max_token_len)
"
```

Charger prompt = **26/48** (the cube's is 16). Also confirm `default_prompt` byte-matches the
dataset's stored task string, so training and deploy can't drift:

```bash
.venv/bin/python -c "
import json; import openpi.training.config as C
task = json.loads(open('$LEROBOT_HOME/trossen_bimanual_charger_plugin_tavla/meta/tasks.jsonl').readline())['task']
print('match:', task == C.get_config('pi0_trossen_charger_plugin_effort_sota').data.default_prompt)
"
```

### 11c. Norm stats — always bound with `--max-frames`

`compute_norm_stats.py` sets `num_batches = num_frames` but consumes only `batch[key][0]`, so with
its `local_batch_size` it loads several samples per *used* frame and cycles the dataset multiple
times. On the charger set an unbounded run was on track for **~4 hours**; it also drove the
sustained MooseFS reads that wedged the pod (§7). Bound it instead:

```bash
export LEROBOT_HOME=/workspace/hf/lerobot HF_LEROBOT_HOME=/workspace/hf/lerobot
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name=pi0_trossen_charger_plugin_effort_sota --max-frames 6000
```

**5.5 minutes**, and `--max-frames` also switches the sampler to `shuffle=True`, so the frames are
drawn randomly across all episodes rather than every Nth frame. 6,000 frames is far more than
enough for per-dimension mean/std/q01/q99 over 14–32 dims. Run it in `tmux` — it is long enough
to lose to a dropped shell.

### 11d. Put the dataset on **local** disk and raise `num_workers`

The single biggest speedup. `TrainConfig.num_workers` defaults to **2**, and `/workspace` is a
MooseFS network mount, so AV1 decode starves the GPU (utilization visibly flapping 0%/100%).
The dataset is only ~1 GB and the container has ~489 GB of local overlay disk:

```bash
mkdir -p /root/hf/lerobot
cp -r /workspace/hf/lerobot/<repo_id> /root/hf/lerobot/

tmux new -d -s train "cd /workspace/EVAN-TA-VLA && \
  export LEROBOT_HOME=/root/hf/lerobot HF_LEROBOT_HOME=/root/hf/lerobot && \
  .venv/bin/python scripts/train.py <config_name> \
    --exp-name run_001 --num-workers 8 --no-wandb-enabled > /tmp/train.log 2>&1"
```

| | dataset on `/workspace`, 2 workers | dataset on local disk, 8 workers |
|---|---|---|
| rate | 4.9 s/it | **2.8–2.9 s/it** |
| GPU util | ~50% (flapping) | ~100% (compute-bound) |
| 30k-step ETA | ~40 h | **~23.5 h** |

Keep **checkpoints** on `/workspace` (the repo lives there, so this is the default) — local disk
does **not** survive a pod restart. Only the dataset copy is disposable; the canonical copy stays
on `/workspace`. Spawning 8 workers adds ~4 min to startup as each re-imports JAX.

### 11e. ⚠️ Check norm stats for near-constant dimensions before deploying

`Normalize` computes `(x - mean) / (std + 1e-6)`. A dimension whose std is effectively zero
therefore divides by ~`1e-6`, and any deploy-time reading that differs from the training constant
produces an enormous normalized value — a **silent, severe out-of-distribution input** that
corrupts the whole state token.

**In the charger dataset the left gripper (index 6) never actuates:** `std = 4.2e-06` for
`state` and `4.7e-05` for `action`. The left arm repositions with a fixed grip while the right
gripper does the ~8 mm grasping. What this means:

- **Training is unaffected.** The model learns to hold that joint constant and will never
  meaningfully command it — un-normalization maps its output back to ≈ the constant.
- **Deployment is not.** The left gripper **must be staged at the training constant (≈0)** before
  the first observation is sent. A real reading of 1 cm normalizes to ~2000, versus the ±1 range
  the model saw in training. The existing all-zeros staging pose (§9b, `env.py`
  `STAGED_POSITIONS`) already satisfies this — do not change it for this task, and re-check it
  for any future dataset.

Run this against any new dataset's norm stats before the first live run:

```bash
.venv/bin/python -c "
import json, numpy as np
p='assets/<config_name>/<repo_id>/norm_stats.json'
ns=json.load(open(p))['norm_stats']
names=['L_j0','L_j1','L_j2','L_j3','L_j4','L_j5','L_grip','R_j0','R_j1','R_j2','R_j3','R_j4','R_j5','R_grip']
for k in ('state','actions'):
    s=np.array(ns[k]['std'])
    print(k, 'zero-std dims:', np.where(s==0)[0].tolist(), '(expect 14..31 = padding only)')
    for i,n in enumerate(names):
        if s[i] < 1e-3: print(f'   !! near-constant: {n} (dim {i}) std={s[i]:.2e}')
"
```

Zero std in dims **14–31 is expected** — that is the pad from 14 real dims up to `action_dim=32`.
Anything flagged in dims 0–13 is a real joint that never moves; note it and stage that joint
precisely at deploy.

### 11f. Reference numbers (A100 80GB PCIe, batch 32)

- **2.8–2.9 s/it** → ~23.5 h for 30k steps. An H100 is roughly 1.5–2× faster.
- Startup to step 0: ~4 min with `pi0_base` cached (first ever run adds a ~12 GiB S3 restore).
- GPU memory: JAX preallocates ~61 GB of 80 GB; batch 32 fits with headroom.
- Checkpoint size: **8.8 GB** (`params` 5.8 GB + `train_state` + embedded `assets/norm_stats.json`).
- Retention: saved every 1,000 steps, `max_to_keep=1` plus `keep_period=5000`, so only the latest
  and multiples of 5,000 survive on disk.
- Loss sanity (charger, SOTA): 0.369 @ step 0 → 0.246 @ 1k → 0.163 @ 4k → 0.052 @ 30k (converged;
  grad_norm settles ~0.20). Flat before step ~1,000 is just the warmup (§7).
- Loss sanity (charger, **base**): 0.141 @ step 0 → 0.041 @ 4k → 0.023 @ 10k → 0.011 @ 22k →
  0.008 @ 30k (grad_norm ~0.12). Checkpoints are **8.9 GB** (slightly under SOTA's 9.0 GB — no
  `effort_proj` and the action projections stay at `action_dim` instead of widening to 46).
- Both charger runs took ~25 h wall-clock for 30k steps on the A100.

> **Do not compare SOTA and base training losses.** The SOTA objective is
> `action_loss + 0.1 * effort_loss` over a 46-wide projection (32 action dims + 14 predicted
> future-torque dims); base optimizes 32 action dims only. Base's much lower number reflects an
> easier objective, not a better policy. Compare the two on **task success on the real arms**
> (grasp stability, over-squeeze, contact timing) — see §6 of `BASE_PI0_ABLATION.md`.

### 11g. Deploying a new policy — what the Trossen box needs

Three artifacts, in order:

1. **Code** — `git pull` on the Trossen box. The new `TrainConfig` lives in
   `src/openpi/training/config.py`; without it `--policy.config <config_name>` will not resolve.
   The tracked `assets/<config>/<repo_id>/norm_stats.json` comes along too. (Serving does not
   strictly need it — `create_trained_policy` deliberately loads norm stats from the
   **checkpoint's** `assets/` dir so inference matches training exactly — but keep them in sync.)
2. **Weights** — rsync the checkpoint (~9 GB). This is a *pull*, so run it **on the Trossen box**,
   not on the pod (the pod has no route to itself and its own key isn't in its `authorized_keys`):
   ```bash
   rsync -rltvP --mkpath --no-owner --no-group --no-perms \
     -e "ssh -p <POD_PORT> -i ~/.ssh/id_ed25519" \
     root@<POD_IP>:/workspace/EVAN-TA-VLA/checkpoints/<config_name>/run_001/29999/ \
     ~/EVAN-TA-VLA/checkpoints/<config_name>/run_001/29999/
   ```
3. **(RTX 5090 deploy box only)** re-apply the JAX/CUDA bump if `uv sync` has been run since:
   `scripts/fix_blackwell_cuda.sh` (or `--check` to verify first). Do **not** run it on the
   training pod — its driver is too old and the CUDA 12.9 libs segfault in cuBLAS (§10a).

> ### The prompt is negotiated automatically — do not pass `--prompt`
> The client sends a `prompt` key on **every** observation and `InjectDefaultPrompt` only fills that
> key when it is **absent**, so a client-side prompt always wins and the server's
> `--default-prompt` is ignored. A hardcoded client prompt therefore fails *silently* when serving
> a different task. The server now advertises its training prompt in the websocket metadata and the
> client adopts it, so **omit `--prompt`** and instead confirm the `Using prompt: '…'` line the
> client logs at startup. Passing `--prompt` is an override; the client warns if it disagrees with
> the server.

#### Charger plug-in — concrete commands

Both charger policies were trained on the **A100 pod at `38.128.233.145:28544`** (pod host/port
change every pod — take them from `$RUNPOD_PUBLIC_IP` / `$RUNPOD_TCP_PORT_22` on the pod).

**Pull the checkpoints — run these ON the Trossen box** (`git pull` first; you need commit
`024fff4` for the base config and `9291151` for the prompt metadata):

```bash
# TA-VLA SOTA (+obs+obj) — 9.0 GB
rsync -rltvP --mkpath --no-owner --no-group --no-perms \
  -e "ssh -p 28544 -i ~/.ssh/id_ed25519" \
  root@38.128.233.145:/workspace/EVAN-TA-VLA/checkpoints/pi0_trossen_charger_plugin_effort_sota/run_001/29999/ \
  ~/EVAN-TA-VLA/checkpoints/pi0_trossen_charger_plugin_effort_sota/run_001/29999/

# Base pi0 ablation (no torque) — 8.9 GB
rsync -rltvP --mkpath --no-owner --no-group --no-perms \
  -e "ssh -p 28544 -i ~/.ssh/id_ed25519" \
  root@38.128.233.145:/workspace/EVAN-TA-VLA/checkpoints/pi0_trossen_charger_plugin_base/run_001/29999/ \
  ~/EVAN-TA-VLA/checkpoints/pi0_trossen_charger_plugin_base/run_001/29999/
```

Verify each with `du -sh` on the target: 9.0 GB / 8.9 GB, `params` 5.8 GB in both.

**Serve** — two processes from `~/EVAN-TA-VLA`. Only the two `--policy.*` flags differ between
the policies; the client command is **identical** for both:

```bash
# Terminal 1 — model server (JAX venv). Pick ONE:

# (a) TA-VLA SOTA
.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_charger_plugin_effort_sota \
  --policy.dir checkpoints/pi0_trossen_charger_plugin_effort_sota/run_001/29999

# (b) base pi0 ablation
.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_charger_plugin_base \
  --policy.dir checkpoints/pi0_trossen_charger_plugin_base/run_001/29999
```

```bash
# Terminal 2 — robot client (lerobot venv), same for both policies.
# Drop --dry-run only after the dry run looks right.
~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 \
  --max-episode-steps 150 --dry-run
```

The client will log
`Using prompt: 'Unplug the charging cube from the power strip, plug it into the adjacent outlet, then turn the power strip switch on'`
— check that before letting it move.

> The client sends `effort` to **both** policies. The base policy ignores it: verified by pinning
> the sampling RNG and confirming with-effort and without-effort inference are bit-identical
> (max diff 0.0). Note `Policy.infer` splits its RNG per call, so two calls on identical inputs
> legitimately differ — pin `policy._rng` before comparing anything.

For a fair A/B, restart Terminal 1 with the other config and rerun the **same** client command with
the same cube/charger placement, seed pose, `--action-horizon` and `--max-episode-steps`.

Then follow the §9 safety ladder (dry run → guarded live with `--max-relative-target 1.0` and
`--action-ema-alpha 0.5`), and honour the near-constant-joint staging check in §11e — for this
dataset the **left gripper must start at ≈0**.
