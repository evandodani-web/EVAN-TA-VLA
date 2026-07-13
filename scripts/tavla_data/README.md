# TA-VLA data conversion (Trossen v3.0 → openpi v2.x)

Converts the Trossen bimanual **lerobot v3.0** dataset (28-dim combined state, 4 cameras)
into the **lerobot v2.x** format that this repo's openpi (lerobot 0.1.0) reads, while
**splitting** the state into a 14-dim position-only `observation.state` and a separate
14-dim `observation.effort` column (required for the TA-VLA history/objective variants).

Two steps because the two lerobot versions can't coexist in one venv:

| Step | Venv | lerobot | Role |
|---|---|---|---|
| 1 | `~/lerobot_trossen/.venv` | 0.5.x | reads v3.0, writes `.npz` intermediate |
| 2 | `~/EVAN-TA-VLA/.venv` | 0.1.0 | reads `.npz`, writes v2.x dataset |

## Step 1 — export to npz (Trossen venv)

```bash
/home/trossen/lerobot_trossen/.venv/bin/python \
  /home/trossen/EVAN-TA-VLA/scripts/tavla_data/export_v30_to_npz.py \
  --src /home/trossen/.cache/huggingface/lerobot/datasets/trossen-bimanual-transfer-cube-external-effort-v2 \
  --out /home/trossen/tavla_intermediate
# add --max-episodes 2 for a quick smoke test
```

## Step 2 — build the v2.x dataset (openpi venv)

```bash
/home/trossen/EVAN-TA-VLA/.venv/bin/python \
  /home/trossen/EVAN-TA-VLA/scripts/tavla_data/npz_to_lerobot_v2.py \
  --intermediate /home/trossen/tavla_intermediate \
  --repo-id trossen_bimanual_transfer_cube_tavla
# output: ~/.cache/huggingface/lerobot/trossen_bimanual_transfer_cube_tavla
```

The `.npz` intermediate can be deleted afterward.

## Result

```
observation.state           (14,)   [L pos(7), R pos(7)]
observation.effort          (14,)   [L ext_eff(7), R ext_eff(7)]
action                      (14,)   joint positions
observation.images.cam_high / cam_left_wrist / cam_right_wrist  (video, 480x640)
```

Task string (verbatim; reuse as `default_prompt` and deploy `--task`):
`Grab and hand over the Rubiks cube to the other arm`

## Transfer to the H100 pod (rsync, no Hugging Face)

```bash
rsync -avP -e "ssh -p <PORT>" \
  ~/.cache/huggingface/lerobot/trossen_bimanual_transfer_cube_tavla/ \
  root@<POD_HOST>:/workspace/hf/lerobot/trossen_bimanual_transfer_cube_tavla/
```

On the pod: `export HF_LEROBOT_HOME=/workspace/hf/lerobot` and use `local_files_only=True`.
