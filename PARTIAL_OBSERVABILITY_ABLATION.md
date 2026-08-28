# Partial-observability ablation (wrist-only) — Runbook

Retrain **both** charger plug-in policies — the torque-aware π0 and the base π0 — with the
external `cam_high` view removed, leaving only the two wrist cameras. Everything else is held
fixed: same dataset, same LoRA setup, same prompt, same 30,000 steps, same normalization.

Combined with the two runs that already exist, this gives a **2×2**:

|                     | with `cam_high`                          | wrist-only                                          |
| ------------------- | ---------------------------------------- | --------------------------------------------------- |
| **TA-VLA (+obs+obj)** | `pi0_trossen_charger_plugin_effort_sota` | `pi0_trossen_charger_plugin_effort_sota_wristonly` |
| **base π0**         | `pi0_trossen_charger_plugin_base`        | `pi0_trossen_charger_plugin_base_wristonly`        |

The question it answers: does the torque signal matter *more* when vision is degraded? Under
partial observability the wrists lose sight of the outlet during the approach, so contact
feedback is the only remaining cue for the insertion — the regime where torque should help most.

Companion to `RUNPOD_SETUP_AND_TRAINING.md` (setup/transfer/perf) and `BASE_PI0_ABLATION.md`
(the base-π0 delta). This file is just the partial-observability delta.

---

## 0. `cam_low` was never in the training data

Worth stating up front, because it is easy to assume otherwise: the rig records **four** cameras,
but `cam_low` is dropped at conversion time by `DEFAULT_CAMERAS` in
`scripts/tavla_data/export_v30_to_npz.py`, and nothing downstream can read it — the repack dict in
`LeRobotTavlaDataConfig.__post_init__` maps only three keys, `TavlaInputs` indexes only those
three, and the deploy client opens only three RealSenses.

So **both existing charger policies were already trained without the low camera.** The only
external view they ever saw is `cam_high`, and that is what this ablation removes.

---

## 1. What makes these runs "wrist-only"

Identical to their full-observation counterparts in every hyperparameter — LoRA
(`gemma_2b_lora` + `gemma_300m_lora`), `freeze_filter`, `ema_decay=None`,
`num_train_steps=30_000`, `pi0_base` weight loader, prompt, `effort_type`, `effort_history`,
dataset — **except** `mask_base_image=True` on the data config.

The masking is done **in the transform, not in the dataset**. `cam_high` is still repacked,
decoded, and handed to `TavlaInputs`, which then replaces it with a zero image and sets its
attention mask to `False`. Two consequences worth understanding:

- **The data pipeline is byte-identical to the full-observation runs.** The only difference
  between the two arms of the ablation is the masking itself, which is a stronger "everything
  else held constant" guarantee than a second dataset conversion would give.
- **There is no speedup.** The loader still AV1-decodes all three streams, and the masked image
  still passes through the ViT — its tokens are merely excluded from attention. Budget the same
  ~25 h per run.

π0 structurally requires all three image slots (`preprocess_observation` raises if any of
`base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb` is missing), so zeros-plus-`False_` is
the correct way to drop a view. It is the same convention `aloha_policy` and `droid_policy`
already use for absent cameras, so `pi0_base`'s pretrained weights see a case they were built for.

---

## 2. Code changes (already applied in this repo)

### 2a. `TavlaInputs` gains `mask_base_image` — `src/openpi/policies/tavla_policy.py`

```python
mask_base_image: bool = False

# in __call__:
images = {
    "base_0_rgb": np.zeros_like(base_image) if self.mask_base_image else base_image,
}
image_masks = {
    "base_0_rgb": np.False_ if self.mask_base_image else np.True_,
}
```

### 2b. `LeRobotTavlaDataConfig` forwards it — `src/openpi/training/config.py`

A `mask_base_image: bool = False` field, passed into `TavlaInputs(...)` in `create()`.
`__post_init__` is deliberately left alone so `cam_high` stays in the repack dict.

**This is why the deploy client needs no changes.** `create_trained_policy` builds its input
stack from `data_config.data_transforms.inputs` — the very same `TavlaInputs` instance the
training loader uses — so the masking is applied server-side at inference automatically. A policy
trained wrist-only cannot accidentally be served with the base view, and the same
`examples/trossen_real` command drives all four policies.

### 2c. Two new `TrainConfig`s — `src/openpi/training/config.py`

`pi0_trossen_charger_plugin_effort_sota_wristonly` and
`pi0_trossen_charger_plugin_base_wristonly`, added after the two full-observation charger
configs. Copies of those blocks with `mask_base_image=True` and nothing else changed.

---

## 3. Norm stats — copied, not recomputed

Norm stats are keyed by config name under `assets/<name>/<repo_id>/`, so the new configs need
their own directories. Their contents hold only `state`, `actions` and `effort` — no camera data
— so copying is both correct and *better* than recomputing: it guarantees bit-identical
normalization to the full-observation runs rather than merely statistically similar.

```bash
cd /workspace/EVAN-TA-VLA
cp -r assets/pi0_trossen_charger_plugin_effort_sota \
      assets/pi0_trossen_charger_plugin_effort_sota_wristonly
cp -r assets/pi0_trossen_charger_plugin_base \
      assets/pi0_trossen_charger_plugin_base_wristonly
```

Already done and committed; the inner `trossen_bimanual_charger_plugin_tavla/` directory name is
unchanged because `repo_id` is the same. Do **not** run `compute_norm_stats.py` for these.

---

## 4. Pre-train smoke check (ON the pod)

Confirm the masking actually reaches the model input before spending ~50 GPU-hours. Note that
uint8 zeros become **−1.0** once `Observation.from_dict` maps images to [−1, 1], so assert
*constant*, not *zero*:

```bash
cd /workspace/EVAN-TA-VLA
export LEROBOT_HOME=/root/hf/lerobot HF_LEROBOT_HOME=/root/hf/lerobot
.venv/bin/python -c "
import dataclasses, numpy as np
import openpi.training.config as C
from openpi.training import data_loader as DL
cfg = dataclasses.replace(C.get_config('pi0_trossen_charger_plugin_effort_sota_wristonly'), batch_size=2)
obs, actions = next(iter(DL.create_data_loader(cfg, shuffle=True, num_batches=1, num_workers=0)))
b = np.asarray(obs.images['base_0_rgb'])
print('base mask all False:', not np.asarray(obs.image_masks['base_0_rgb']).any())
print('base image constant:', b.min() == b.max(), float(b.min()))
print('wrists unmasked:', bool(np.asarray(obs.image_masks['left_wrist_0_rgb']).all()),
                          bool(np.asarray(obs.image_masks['right_wrist_0_rgb']).all()))
print('wrist image varies:', float(np.asarray(obs.images['left_wrist_0_rgb']).std()) > 0.01)
print('effort:', tuple(np.asarray(obs.effort).shape))
"
```

Expect `base mask all False: True`, `base image constant: True -1.0`, both wrists `True`,
`wrist image varies: True`, and `effort: (2, 60, 14)`.

Run the same check for `pi0_trossen_charger_plugin_base_wristonly` — there `obs.effort` should be
`None` instead (replace the last print with `print('effort is None:', obs.effort is None)`).

As a control, the *unmasked* config must still report `base mask all False: False`:

```bash
.venv/bin/python -c "
import openpi.training.config as C
for n in ('pi0_trossen_charger_plugin_effort_sota', 'pi0_trossen_charger_plugin_effort_sota_wristonly',
          'pi0_trossen_charger_plugin_base', 'pi0_trossen_charger_plugin_base_wristonly'):
    cfg = C.get_config(n)
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    print(f'{n:52s} mask={cfg.data.mask_base_image!s:5s} '
          f'effort_dim_in={getattr(cfg.model, \"effort_dim_in\", None)} '
          f'norm_stats={None if dc.norm_stats is None else sorted(dc.norm_stats)}')
"
```

Expect `mask=False` / `True` alternating down the list, `effort_dim_in=140` for the two SOTA
configs and `None` for the two base ones, and norm-stat keys `['actions', 'effort', 'state']` for
SOTA versus `['actions', 'state']` for base.

> `effort_dim_in` is **not** a declared `Pi0Config` field — `TrainConfig.__post_init__` injects it
> only when `effort_history` is non-empty. On the base configs the attribute genuinely does not
> exist, hence the `getattr` above; reading it directly raises `AttributeError`. Nothing needs it
> there, since `effort_type=NO` means `effort_proj_in` is never built.

---

## 5. Train (ON the pod, inside tmux)

Apply the §11d performance rules from `RUNPOD_SETUP_AND_TRAINING.md` — dataset on **local** disk
and `--num-workers 8`, which is what cut the earlier charger runs from ~40 h to ~23.5 h. The
dataset copy is shared by all four charger configs, so if it is already at `/root/hf/lerobot`
from the previous runs, skip the `cp`.

```bash
mkdir -p /root/hf/lerobot
cp -r /workspace/hf/lerobot/trossen_bimanual_charger_plugin_tavla /root/hf/lerobot/

# TA-VLA SOTA, wrist-only
tmux new -d -s train_sota_wo "cd /workspace/EVAN-TA-VLA && \
  export LEROBOT_HOME=/root/hf/lerobot HF_LEROBOT_HOME=/root/hf/lerobot && \
  .venv/bin/python scripts/train.py pi0_trossen_charger_plugin_effort_sota_wristonly \
    --exp-name run_001 --num-workers 8 --wandb-enabled > /tmp/train_sota_wristonly.log 2>&1"

# base pi0, wrist-only
tmux new -d -s train_base_wo "cd /workspace/EVAN-TA-VLA && \
  export LEROBOT_HOME=/root/hf/lerobot HF_LEROBOT_HOME=/root/hf/lerobot && \
  .venv/bin/python scripts/train.py pi0_trossen_charger_plugin_base_wristonly \
    --exp-name run_001 --num-workers 8 --wandb-enabled > /tmp/train_base_wristonly.log 2>&1"
```

Run these **sequentially, not concurrently**, on a single GPU: JAX preallocates ~61 GB of 80 GB,
so two jobs will not co-exist. ~25 h each, so ~50 h total.

Checkpoints land in `checkpoints/<config_name>/run_001/`, saved every 1,000 steps, kept at
multiples of 5,000, final at `29999`. Keep them on `/workspace` — local disk does not survive a
pod restart.

### What the loss should look like

Expect **higher** final loss than the corresponding full-observation run — that is the point of
the ablation, not a bug. For reference, the full-observation runs finished at 0.052 (SOTA) and
0.008 (base).

> **Only compare like with like.** SOTA-vs-base losses are not comparable in either column (the
> SOTA objective is `action_loss + 0.1 * effort_loss` over a 46-wide projection; base optimizes 32
> action dims). The meaningful comparisons here are *down* each column of the 2×2, and ultimately
> task success on the real arms.

---

## 6. Deploy — no client changes

Pull the checkpoints as in §11g of `RUNPOD_SETUP_AND_TRAINING.md`, then serve with the matching
config name:

```bash
# Terminal 1 — pick ONE
.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_charger_plugin_effort_sota_wristonly \
  --policy.dir checkpoints/pi0_trossen_charger_plugin_effort_sota_wristonly/run_001/29999

.venv/bin/python scripts/serve_policy.py --port 8000 \
  policy:checkpoint --policy.config pi0_trossen_charger_plugin_base_wristonly \
  --policy.dir checkpoints/pi0_trossen_charger_plugin_base_wristonly/run_001/29999
```

```bash
# Terminal 2 — identical for all four policies
~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
  --host localhost --port 8000 --action-horizon 25 \
  --max-episode-steps 150 --dry-run
```

> **Keep the high camera plugged in.** The client still opens three RealSenses and `TavlaInputs`
> still indexes `in_images["cam_high"]` — it just zeroes the frames server-side. Unplugging it
> will fail at `connect()`, before the policy is ever consulted.

Everything else is unchanged: omit `--prompt` (the server advertises it and the client adopts it),
follow the §9 safety ladder, and stage the **left gripper at ≈0** — its norm-stat std is 4.2e−06,
so a small physical offset becomes a huge out-of-distribution input (§11e).

---

## 7. Evaluating the 2×2

For each of the four policies, restart Terminal 1 with that config's `--policy.config` and
`--policy.dir` and run the *same* client command with the same charger/outlet placement, seed
pose, `--action-horizon` and `--max-episode-steps`. Score the sub-goals separately, since partial
observability and torque should affect different ones:

| Sub-goal | Mostly limited by |
| --- | --- |
| Reach + grasp the charging cube | vision (expect the wrist-only runs to degrade here) |
| Unplug (extraction) | contact/torque |
| Align to the adjacent outlet | vision (the hardest step without `cam_high`) |
| Insert (plug-in) | contact/torque — **the cell where TA-VLA should show the largest lift** |
| Flip the power strip switch | vision + contact |

The headline number for the paper claim is the *difference of differences*: TA-VLA's lift over
base π0 in the wrist-only column minus its lift in the full-observation column. A positive value
is the evidence that torque substitutes for missing visual information.
