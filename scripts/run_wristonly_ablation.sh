#!/usr/bin/env bash
# Chained runner for the wrist-only partial-observability ablation (charger plug-in).
#
#   1. wait for the dataset to arrive at $STAGE_SRC and stop changing
#   2. copy it to local disk (the /workspace MooseFS mount starves the GPU -- see §11d
#      of RUNPOD_SETUP_AND_TRAINING.md: 4.9 s/it -> 2.8 s/it)
#   3. train the torque-aware policy to 30k steps
#   4. buffer, so VRAM is fully released and the last checkpoint is flushed
#   5. train the base pi0 policy to 30k steps
#
# The two runs go sequentially on purpose: JAX preallocates ~61 GB of the 80 GB card,
# so they cannot co-exist.
#
# Usage:  tmux new -d -s ablation 'bash scripts/run_wristonly_ablation.sh'
# Logs:   /workspace/train_logs/   (on the network volume, so they survive a pod restart)

set -uo pipefail

REPO=/workspace/EVAN-TA-VLA
REPO_ID=trossen_bimanual_charger_plugin_tavla
STAGE_SRC=/workspace/hf/lerobot/$REPO_ID     # where the rsync from the Trossen box lands
LOCAL_HOME=/root/hf/lerobot                  # fast local overlay disk
LOCAL_DS=$LOCAL_HOME/$REPO_ID
LOG_DIR=/workspace/train_logs

SOTA_CONFIG=pi0_trossen_charger_plugin_effort_sota_wristonly
BASE_CONFIG=pi0_trossen_charger_plugin_base_wristonly
EXP_NAME=run_001

EXPECT_EPISODES=56
EXPECT_FRAMES=40940
BUFFER_SECONDS=600                           # 10 min between runs
EARLY_FAILURE_SECONDS=1800                   # a run dying inside 30 min is structural

mkdir -p "$LOG_DIR"
MAIN_LOG=$LOG_DIR/ablation_chain.log

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%SZ')] $*" | tee -a "$MAIN_LOG"; }

# Optional secrets file, kept outside the repo so the key is never committed.
# Resolved at the start of each run rather than here, so it can be dropped in at any
# point before training actually begins without restarting the chain.
WANDB_ENV_FILE=/workspace/.wandb_env

resolve_wandb() {
  # shellcheck disable=SC1090
  [ -f "$WANDB_ENV_FILE" ] && . "$WANDB_ENV_FILE"
  if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_FLAG=--wandb-enabled
    log "W&B key found -> logging enabled."
  else
    WANDB_FLAG=--no-wandb-enabled
    log "No W&B key (looked in the environment and $WANDB_ENV_FILE) -> --no-wandb-enabled."
  fi
}

# ---------------------------------------------------------------- 1. wait for data

wait_for_dataset() {
  if [ -d "$LOCAL_DS" ]; then
    log "Dataset already on local disk at $LOCAL_DS -- skipping wait and copy."
    return 0
  fi

  log "Waiting for the dataset to appear at $STAGE_SRC ..."
  log "  (run the rsync from the Trossen box; this script picks it up automatically)"
  until [ -f "$STAGE_SRC/meta/info.json" ]; do sleep 60; done
  log "meta/info.json found. Waiting for the transfer to settle ..."

  # rsync writes .~tmp~ files and grows the tree; require two identical size
  # readings 60 s apart with no temp files left behind.
  local prev=-1 cur
  while true; do
    sleep 60
    if find "$STAGE_SRC" -name '.*.~tmp~' -o -name '*.partial' 2>/dev/null | grep -q .; then
      log "  ...rsync temp files still present"
      prev=-1
      continue
    fi
    cur=$(du -sb "$STAGE_SRC" 2>/dev/null | cut -f1)
    if [ -z "$cur" ]; then prev=-1; continue; fi
    if [ "$cur" = "$prev" ]; then
      log "Transfer settled at $(numfmt --to=iec "$cur")."
      break
    fi
    log "  ...still growing ($(numfmt --to=iec "${cur:-0}"))"
    prev=$cur
  done

  # Sanity-check against the known charger dataset shape before spending ~50 GPU-hours.
  local eps frames
  eps=$(python3 -c "import json;print(json.load(open('$STAGE_SRC/meta/info.json'))['total_episodes'])" 2>/dev/null)
  frames=$(python3 -c "import json;print(json.load(open('$STAGE_SRC/meta/info.json'))['total_frames'])" 2>/dev/null)
  log "Dataset reports episodes=$eps frames=$frames (expected $EXPECT_EPISODES / $EXPECT_FRAMES)."
  if [ "$eps" != "$EXPECT_EPISODES" ] || [ "$frames" != "$EXPECT_FRAMES" ]; then
    log "WARNING: does not match the expected charger dataset. Continuing anyway --"
    log "         check this before trusting the results."
  fi

  log "Copying dataset to local disk ($LOCAL_DS) for the ~1.7x throughput win ..."
  mkdir -p "$LOCAL_HOME"
  if cp -r "$STAGE_SRC" "$LOCAL_HOME/"; then
    log "Copy complete."
  else
    log "ERROR: copy to local disk failed; aborting."
    exit 1
  fi
}

# ---------------------------------------------------------------- 2. train one config

wait_for_free_gpu() {
  local used
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-99999}" -lt 2000 ]; then return 0; fi
    log "  ...GPU still holding ${used} MiB, waiting"
    sleep 30
  done
  log "WARNING: GPU still busy after 30 min; starting anyway."
}

run_training() {
  local config=$1 logfile=$2 start_ts end_ts rc

  resolve_wandb
  wait_for_free_gpu
  log "=== START $config -> $logfile"
  start_ts=$(date +%s)

  cd "$REPO" || exit 1
  LEROBOT_HOME=$LOCAL_HOME HF_LEROBOT_HOME=$LOCAL_HOME \
    .venv/bin/python scripts/train.py "$config" \
      --exp-name "$EXP_NAME" \
      --num-workers 8 \
      $WANDB_FLAG >"$logfile" 2>&1
  rc=$?

  end_ts=$(date +%s)
  local elapsed=$(( end_ts - start_ts ))
  log "=== END   $config  exit=$rc  elapsed=$(( elapsed / 3600 ))h$(( (elapsed % 3600) / 60 ))m"

  if [ $rc -ne 0 ] && [ $elapsed -lt $EARLY_FAILURE_SECONDS ]; then
    log "ERROR: $config died after only $(( elapsed / 60 )) min -- that is a structural"
    log "       problem, not a transient one. Aborting the chain so the second run does"
    log "       not fail the same way. Tail of the log:"
    tail -30 "$logfile" | tee -a "$MAIN_LOG"
    return 2
  fi
  return $rc
}

# ---------------------------------------------------------------- 3. the chain

log "########## wrist-only ablation chain starting (pod $(hostname)) ##########"
wait_for_dataset

run_training "$SOTA_CONFIG" "$LOG_DIR/train_${SOTA_CONFIG}.log"
sota_rc=$?
if [ $sota_rc -eq 2 ]; then
  log "Chain aborted. Nothing else will run."
  exit 1
fi
[ $sota_rc -ne 0 ] && log "NOTE: $SOTA_CONFIG exited $sota_rc late in the run; continuing to the base run."

log "Buffering ${BUFFER_SECONDS}s before the base run (VRAM release + checkpoint flush) ..."
sleep $BUFFER_SECONDS

run_training "$BASE_CONFIG" "$LOG_DIR/train_${BASE_CONFIG}.log"
base_rc=$?

log "########## chain finished: $SOTA_CONFIG=$sota_rc  $BASE_CONFIG=$base_rc ##########"
log "Checkpoints: $REPO/checkpoints/{$SOTA_CONFIG,$BASE_CONFIG}/$EXP_NAME/29999"
