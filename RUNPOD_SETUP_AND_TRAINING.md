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
matters for Cursor Remote-SSH and rsync. `:2022` is typically for Eternal Terminal (optional).

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
.venv/bin/python scripts/train.py \
  --config-name=pi0_trossen_transfer_effort_sota \
  --exp-name=run_001
# add --wandb_enabled=True to log to Weights & Biases.
# For W&B: do NOT use `wandb login` (the interactive prompt rejects new-format
# `wandb_v1_...` keys). Instead set the key as an env var:
#   echo 'export WANDB_API_KEY=<your wandb_v1_ key>' >> ~/.bashrc && source ~/.bashrc
```

Ablation baseline (pure-obs DePost, `EXPERT`) — same command, different config:
```bash
.venv/bin/python scripts/train.py \
  --config-name=pi0_trossen_transfer_effort_expert \
  --exp-name=run_001
```

Notes:
- `--exp-name` is **required**; names the checkpoint dir
  `checkpoints/<config_name>/<exp_name>/`.
- `num_train_steps=30_000`; checkpoints every 1,000 steps, kept at multiples of 5,000.
- Resume an interrupted run: add `--resume=True` (mutually exclusive with `--overwrite=True`).
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

---

## 8. Next: deployment (Phase 3, later)
After a checkpoint exists: serve it with openpi's websocket policy server
(`scripts/serve_policy.py`) and write a Trossen client (10-frame effort history buffer at the
training offsets, EMA + `max_relative_target` + fps pacing). See Phase 3 in
`TROSSEN_TAVLA_FINDINGS_AND_PLAN.md`.
