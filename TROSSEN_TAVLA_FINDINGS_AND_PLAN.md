# Torque-Aware π0 (TA-VLA) on Trossen — Findings & Plan

Working notes for training a **torque-aware π0** on the Trossen bimanual WidowXAI setup for
a contact-rich handover ("Grab and hand over the Rubik's cube to the other arm"), using the
**TA-VLA** reference implementation in this repo (`EVAN-TA-VLA`, built on **openpi** / JAX).

Goal: reproduce the paper's SOTA observation+objective variant (**π0 + obs + obj**,
`EffortType.EXPERT_HIS_C_FUT`) on our own dataset, then deploy on the real robot.

---

## 1. What this repo is

- `EVAN-TA-VLA` is the **official TA-VLA implementation**, branched from **openpi**
  (Physical Intelligence, JAX/Flax/nnx) at commit `cd82848`. It is **not** the `lerobot`
  (PyTorch) codebase.
- The entire TA-VLA design space is already implemented and selected via an enum:
  `src/openpi/shared/effort_type.py` (`EffortType`).
- π0 state/effort injection lives in `src/openpi/models/pi0.py`
  (`embed_prefix`, `embed_suffix`, `_process_effort_tokens`, `compute_loss`).

### `EffortType` → paper method map

| `EffortType` | Paper method | Where effort enters |
|---|---|---|
| `NO` | baseline | not used (still normed) |
| `STATE` | DePre | overwritten into `state[-14:]` → existing `state_proj` |
| `LLM` / `LLM_HIS_*` | Enc / Enc-1 / Enc-H | token(s) in the VLM prefix (worst; avoid) |
| `EXPERT` | DePost | separate token to the action expert (best pure-obs) |
| `EXPERT_HIS_C` / `EXPERT_HIS_T` | Dec-1 / Dec-H | history → token(s) to action expert |
| `EXPERT_FUT`, **`EXPERT_HIS_C_FUT`** | +obj, **+obs+obj (SOTA)** | predict future torque alongside actions |

The `*_FUT` variants widen `action_in_proj`/`action_out_proj` to `action_dim + effort_dim`
and train with `action_loss + 0.1 * effort_loss` (paper's β = 0.1). This is already in
`compute_loss` (the "add weighted action-torque loss" commit).

---

## 2. Our dataset

Path: `~/.cache/huggingface/lerobot/datasets/trossen-bimanual-transfer-cube-external-effort-v2`

- **50 episodes, 39,650 frames, 30 fps**, single task:
  `"Grab and hand over the Rubik's cube to the other arm"`.
- Size: **~1.2 GB** (AV1 video ~1.2 GB; parquet ~6 MB; meta ~0.5 MB).
- **`observation.state` = 28-dim, COMBINED**:
  `[L pos(7), L ext_eff(7), R pos(7), R ext_eff(7)]`.
- **`action` = 14-dim** joint positions (7/arm = 6 joints + carriage gripper).
- **Cameras = 4**: `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist`.
  TA-VLA's `TavlaInputs` uses exactly `cam_high` (base), `cam_left_wrist`, `cam_right_wrist`;
  `cam_low` is ignored. ✅ the 3 we need are present.
- Effort sanity: arm joints O(0.1–1) Nm; gripper effort is noisy / ~no contact signal
  (paper A.2.4). Harmless to keep.
- Format: lerobot **`codebase_version: v3.0`**
  (`data/chunk-000/file-000.parquet`, `meta/episodes/*.parquet`).

---

## 3. The two mismatches that drive the plan

1. **State layout.** The SOTA method needs a **separate `observation.effort` column** with
   *history* (and *future* frames for the objective), sampled via `delta_timestamps`.
   Our effort is baked into the 28-dim state → no `observation.effort` column exists, so
   history/future can't be sampled.
   - `scripts/strip_state_effort.py` (in `lerobot_trossen`) does the *opposite* (drops effort).
   - We need a **split**: `observation.state` → 14 positions, `observation.effort` → 14 ext_eff.
   - TA-VLA expects: `observation.state` = 14, `observation.effort` = `[history, 14]`,
     `action` = 14 (padded to `action_dim = 32` internally; `TavlaOutputs` returns first 14).

2. **Dataset version.** openpi TA-VLA **pins lerobot `0.1.0`** (`rev 6674e368`, old
   `lerobot.common.datasets…` API, `local_files_only` + `delta_timestamps`) → reads **v2.x**
   datasets. Our dataset is **v3.0** and will not load as-is. It must be re-materialized to v2.x.
   - The openpi repo `.venv` is present and functional (`uv sync` was already run).

---

## 4. Deployment reference (`symbiotic/symbiotic/rl/deploy.py`)

- It is a **TD-MPC2** deploy loop (Hydra), *not* a pi0/openpi path — but it is the reference
  for **safe Trossen control**: builds `lerobot_trossen` followers (`make_follower`), reads
  `get_observation()`, translates via `RobotController` (`preprocess_obs` → `agent.act` →
  `postprocess_action`), applies **EMA smoothing + `max_relative_target` clamp + TCP-Z safety**,
  paces at `fps` with `precise_sleep`, handles Trossen units (rad arm / m gripper).
- The Trossen follower already exposes effort via `include_external_effort=True`
  (`BiWidowXAIFollowerRobotConfig`) + `enrich_observation_velocity_load`.
- π0 is JAX → serve with openpi's **websocket policy server** (`scripts/serve_policy.py` +
  `packages/openpi-client`) and a custom Trossen client. lerobot's `async_inference.policy_server`
  cannot host an openpi checkpoint, so it is not used here.
- `examples/aloha_real` is the closest existing openpi client: it already maintains an
  **effort history buffer** for `*_HIS_*` policies — adapt its logic to the `lerobot_trossen`
  follower API.

---

## 5. Plan

### Phase 0 — Environment (on the H100 box)
- `uv sync` the openpi repo; verify JAX sees the GPU.
- Fetch `pi0_base` weights: `s3://openpi-assets/checkpoints/pi0_base/params`.

### Phase 1 — Data conversion (main prep) — v3.0 → v2.x + split state ✅ COMPLETE
**Decision: convert on this machine, then rsync the finished v2.x dataset to the pod.**
Implemented as a two-step **npz bridge** (no new deps; both venvs already have numpy+cv2):

Environment facts (verified):
- `~/lerobot_trossen/.venv` → lerobot **0.5.2** (reads v3.0). Has cv2, PIL, torchvision, numpy.
- `~/EVAN-TA-VLA/.venv` → lerobot **0.1.0** (writes v2.x via `lerobot.common.datasets`).
  Has cv2, h5py, numpy, torch.
- `LEROBOT_HOME = ~/.cache/huggingface/lerobot`; ~650 GB free disk.

Scripts (`scripts/tavla_data/`) — see `scripts/tavla_data/README.md` for exact commands:
1. **`export_v30_to_npz.py`** (run with the **`lerobot_trossen` venv**): reads the v3.0
   dataset via `LeRobotDataset` (decodes AV1 → RGB), splits `observation.state`(28) into
   `qpos`(14 = L pos + R pos) and `effort`(14 = L ext_eff + R ext_eff), and writes one
   `episode_XXXXXX.npz` per episode holding `qpos`, `effort`, `action`, `task`, and the 3
   cameras as JPEG-encoded byte arrays (q95).
2. **`npz_to_lerobot_v2.py`** (run with the **openpi venv**): reads the npz files and builds a
   v2.x `LeRobotDataset` with `observation.state`(14), `observation.effort`(14),
   `action`(14), and `observation.images.{cam_high,cam_left_wrist,cam_right_wrist}` at fps=30.

**Conversion output (verified, Jul 13 2026):**
- Path: `~/.cache/huggingface/lerobot/trossen_bimanual_transfer_cube_tavla`
- **50 episodes, 39,650 frames, fps=30, codebase v2.0, ~987 MB**
- Features confirmed: `observation.state (14)`, `observation.effort (14)`, `action (14)`,
  three `video (3,480,640)` cameras — exactly what `TavlaInputs` expects.
- Task string (verbatim, no apostrophe): `Grab and hand over the Rubiks cube to the other arm`
- Effort `delta_timestamps` query (10-frame history + 50-frame future) returns shape **(60, 14)** ✅
- Norm stats computed during `consolidate()` and baked into the dataset.
- Intermediate npz files retained at `/home/trossen/tavla_intermediate` (safe to delete).
- **Original v3.0 dataset untouched.**

Note on gripper dims: effort indices 6 and 13 (gripper carriages) show large mean offsets
(~−20, ~−36 Nm) vs O(0.01–0.5) for arm joints. This is the expected "gripper effort ≈ gravity
bias, no contact signal" behaviour (paper A.2.4) — normalization handles it; don't expect
contact information from those two dims.

### Data transfer to RunPod — rsync over SSH (NOT Hugging Face)
Sensitive IP → do **not** push to the Hub. Put data on a **persistent Network Volume**
(`/workspace`) on the pod, then:
```bash
rsync -avP -e "ssh -p <PORT>" \
  ~/.cache/huggingface/lerobot/trossen_bimanual_transfer_cube_tavla/ \
  root@<POD_HOST>:/workspace/hf/lerobot/trossen_bimanual_transfer_cube_tavla/
```
(`-z` omitted: video is already AV1-compressed.) On the pod, set
`export HF_LEROBOT_HOME=/workspace/hf/lerobot` (or symlink) so the loader finds it, and use
`local_files_only=True` in the data config.

### Phase 2 — Training config (SOTA variant) ✅ COMPLETE (Jul 13 2026)
Two `TrainConfig`s were added to `src/openpi/training/config.py` (right after the
`pi0_lora_effort_history` template), plus norm stats computed locally. Training itself is
**not** run here — that is done by the user (on the H100 pod).

**1. SOTA config — `pi0_trossen_transfer_effort_sota`** (paper "+obs+obj"):
  - `model = Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",`
    `effort_type=EffortType.EXPERT_HIS_C_FUT, effort_dim=14)`.
  - `data = LeRobotTavlaDataConfig(repo_id="trossen_bimanual_transfer_cube_tavla",`
    `effort_history=tuple(4*i-36 for i in range(10)),`  ← offsets `(-36…0)` step 4 = 10 frames / ~1.2 s @ 30 fps
    `default_prompt="Grab and hand over the Rubik's cube to the other arm",`
    `base_config=DataConfig(local_files_only=True))`.
  - `weight_loader=CheckpointWeightLoader(pi0_base)`, `num_train_steps=30_000`,
    LoRA `freeze_filter`, `ema_decay=None` — identical to the existing effort configs.

**2. Ablation config — `pi0_trossen_transfer_effort_expert`** (pure-obs "DePost"):
  - Same as above but `effort_type=EffortType.EXPERT` and `effort_history=(0,)` (current effort
    only, no history, no future objective). Directly comparable to the SOTA config — a single
    enum + history change — for quantifying the lift from history+objective.

**Why these settings (verified against the code, not assumed):**
- `EXPERT_HIS_C_FUT` is the *only* enum value that gives the paper's SOTA "+obs+obj":
  history torque → one concatenated token to the action expert **and** future torque predicted
  alongside actions. The shipped `pi0_lora_effort*` configs use `EXPERT`/`EXPERT_HIS_C` (neither
  is SOTA) and point at placeholder `repo_id="org/repo"` / `default_prompt="do something"`.
- The future-torque objective is **fully automatic** given the enum (traced end-to-end):
  - `data_loader.create_dataset` (lines 110–113) appends `+1…+action_horizon` future frames to
    the `observation.effort` `delta_timestamps` when the type is a `*_FUT` variant → the column
    comes back as `[len(effort_history)+action_horizon, 14] = [10+50, 14] = [60,14]`.
  - `pi0.compute_loss` (lines 334–339) peels the last `action_horizon` frames off as the
    future-effort *target* (concatenated onto actions) and keeps the first `len(effort_history)`
    frames as the history *input*.
  - `_process_effort_tokens` (EXPERT_HIS_C_FUT branch) flattens the `[10,14]` history → `[140]`,
    matching `effort_dim_in`.
  - `action_in_proj`/`action_out_proj` widen to `action_dim + effort_dim = 32+14 = 46`; loss is
    `action_loss + 0.1*effort_loss`.
- `TrainConfig.__post_init__` auto-sets `effort_dim_in = effort_dim * len(effort_history)`:
  **140** for SOTA (`14*10`), **14** for the ablation (`14*1`). `effort_dim_in` is not a declared
  `Pi0Config` field — it is injected here — so the model must be built via a `TrainConfig` (both
  of ours are). Confirmed by construction test: SOTA → `effort_dim_in=140`, EXPERT → `14`.
- `action_horizon` is kept at the default **50** (per decision): it silently controls how many
  future effort frames are sampled/sliced. Do **not** put future offsets in `effort_history` — it
  holds the 10 history offsets only; `future_steps` is taken from `actions.shape[1]`.
- `default_prompt` set (not `None`) ⇒ `prompt_from_task=False` ⇒ the dataset's task string is
  ignored and this exact prompt is injected at train time. It **must** byte-match the Phase 3
  deploy prompt. (Note: this apostrophe form differs from the dataset's stored task string
  `"…Rubiks cube…"`, which is now irrelevant since we override it.)

**Norm stats (computed locally, Jul 13 2026):**
- `scripts/compute_norm_stats.py --config-name=<config>` was run for **both** configs over all
  **39,650 frames**. Output: `assets/<config_name>/trossen_bimanual_transfer_cube_tavla/norm_stats.json`.
- Verified keys/shapes: `state (32)`, `actions (32)` (padded to `action_dim`), `effort (14)`.
  The `effort` key is only computed because `effort_history` is set (see `compute_norm_stats.py`
  lines 82–83). These assets must be **rsync'd to the pod alongside the dataset** (or recomputed
  there) — norm stats are keyed by config name under `assets/`.
- One-time setup done: the PaliGemma tokenizer is fetched by `config.data.create(...)` (cached in
  `~/.cache/openpi`); it is *not* used in the norm-stats pass itself but is constructed eagerly.

**Remaining before training (user runs on the pod):**
- `python scripts/train.py --config-name=pi0_trossen_transfer_effort_sota --exp-name=<run>`
  (`--wandb_enabled` optional). `exp_name` is required. Ablation: swap the config name.

### Phase 3 — Deployment on Trossen
- Run `scripts/serve_policy.py` with the trained checkpoint (websocket server).
- Write a Trossen client (structure from `symbiotic/rl/deploy.py` + `examples/aloha_real`):
  - `bi_widowxai_follower_robot` with `include_external_effort=True` + 3 cameras;
  - maintain a **10-frame effort history buffer at the training offsets** (`-36…0` step 4);
  - assemble the `TavlaInputs` obs (`state`(14), `effort`(10,14), 3 images,
    prompt = exact training task string); query server → 14-dim action chunk;
  - stream to arms with **EMA + `max_relative_target` + fps pacing** for safety.
- Evaluate the handover contact moment vs a position-only baseline to quantify the lift.

### Open items
- ~~Conversion route~~ — done, npz bridge.
- ~~Phase 2: add `EXPERT_HIS_C_FUT` (SOTA) + `EXPERT` (ablation) train configs to `config.py`~~
  — done (`pi0_trossen_transfer_effort_sota`, `pi0_trossen_transfer_effort_expert`);
  norm stats computed locally for both.
- Transfer to pod: rsync the dataset **and** `assets/pi0_trossen_transfer_effort_sota/` +
  `assets/pi0_trossen_transfer_effort_expert/` (norm stats), then `scripts/train.py`.
- Phase 3: deploy — openpi websocket server + custom Trossen client (recommended).

---

## 6. Key file references
- `src/openpi/shared/effort_type.py` — the design-space enum.
- `src/openpi/models/pi0.py` — `embed_prefix`/`embed_suffix`/`_process_effort_tokens`/`compute_loss`.
- `src/openpi/models/model.py` — `Observation.effort`, `preprocess_observation` (`STATE` overwrite).
- `src/openpi/policies/tavla_policy.py` — `TavlaInputs`/`TavlaOutputs` (3 cams, pad→32, return 14).
- `src/openpi/training/config.py` — `LeRobotTavlaDataConfig`, `effort_history`,
  `effort_dim_in` auto-compute, example configs `pi0_lora_effort` / `pi0_lora_effort_history`.
- `src/openpi/training/data_loader.py` — `delta_timestamps` for effort history + future.
- `scripts/compute_norm_stats.py` — effort norm stats.
- `examples/aloha_real/convert_aloha_data_to_lerobot.py` — HDF5 → v2.x lerobot (has_effort).
- `examples/aloha_real/*` — websocket client + effort history buffer pattern.
- `~/lerobot_trossen/scripts/strip_state_effort.py` — reference for slicing state parquets.
- `~/symbiotic/symbiotic/rl/deploy.py` — Trossen control-loop / safety reference.
