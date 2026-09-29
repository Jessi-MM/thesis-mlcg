#!/usr/bin/env bash
# ============================================================================
# Launcher for the BBA SchNet training — tmux version.
#
# Starts the training inside a tmux session called "bbatrain", so you can
# detach, close VS Code, go home, and attach again later to the SAME terminal.
#
#   ./run_train_tmux.sh        start (or resume) the run
#   LOG=1 ./run_train_tmux.sh  same, but also write a plain-text log file
#   tmux attach -t bbatrain    go look at it   (detach again with Ctrl-b then d)
#   tmux ls                    list your sessions
#
# Difference from run_train.sh (the nohup version): that one detaches the
# process and writes to a log file you can only read. This one keeps a real
# terminal alive, so you get the live progress bar back and can scroll.
#
# The progress bar is left ON here on purpose. Do NOT pipe this into a log
# file: the bar redraws every batch, which over 1000 epochs would write an
# absurdly large file. TensorBoard is the permanent record of the losses.
# ============================================================================
set -euo pipefail

# The ":-" means "use this unless already set in the environment", so the
# script can be smoke-tested without touching the real run, e.g.
#   SESSION=test RUN_DIR=/srv/data/itzeljem79/mlcg-tk/bba/train/_t ./run_train_tmux.sh
SESSION="${SESSION:-bbatrain}"
VENV="${VENV:-/storage/mi/itzeljem79/Projects/mlcg/.venv}"
CONFIG="${CONFIG:-/storage/mi/itzeljem79/Projects/thesis-mlcg/training/bba/training.yaml}"
RUN_DIR="${RUN_DIR:-/srv/data/itzeljem79/mlcg-tk/bba/train/schnet_badn_v1}"
EXTRA="${EXTRA:-}"   # extra flags passed straight to mlcg-train_h5
LOG="${LOG:-0}"      # LOG=1 -> also write a plain-text log file (see below)

# Already running? Don't start a second one on the same output directory.
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "A tmux session '${SESSION}' already exists."
  echo "Look at it with:   tmux attach -t ${SESSION}"
  echo "If the run inside it has finished and you want to start over:"
  echo "                   tmux kill-session -t ${SESSION}"
  exit 1
fi

mkdir -p "${RUN_DIR}"

if [ -f "${RUN_DIR}/ckpt/last.ckpt" ]; then
  echo "Found ${RUN_DIR}/ckpt/last.ckpt -> RESUMING from it."
  echo "(that is what 'ckpt_path: last' in the yaml does)"
else
  echo "No checkpoint found -> starting a FRESH run."
fi

# Generous scrollback, so a crash traceback is still there days later.
tmux set-option -g history-limit 50000 2>/dev/null || true

# Two modes:
#   default  live progress bar in tmux, no log file. The bar redraws every
#            batch, so writing it to disk over 1000 epochs is not an option.
#   LOG=1    progress bar OFF, everything teed to a text file. Use this if you
#            want a record that survives the tmux server being restarted.
#            TensorBoard records the losses either way; the log is really for
#            keeping a crash traceback.
if [ "${LOG}" = "1" ]; then
  LOGFILE="${RUN_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
  TRAIN_CMD="mlcg-train_h5 fit --config '${CONFIG}' --trainer.enable_progress_bar=false ${EXTRA} 2>&1 | tee '${LOGFILE}'"
  echo "log    : ${LOGFILE}   (progress bar disabled in this mode)"
else
  LOGFILE=""
  TRAIN_CMD="mlcg-train_h5 fit --config '${CONFIG}' ${EXTRA}"
fi

# 'pipefail' so the reported exit code is the TRAINING's, not tee's.
# The trailing 'exec bash' keeps the pane alive after training ends or
# crashes, so the last screen of output is still readable when you attach.
tmux new-session -d -s "${SESSION}" \
  "set -o pipefail; \
   source '${VENV}/bin/activate' && \
   echo 'config : ${CONFIG}' && \
   echo 'outputs: ${RUN_DIR}' && \
   echo && \
   ${TRAIN_CMD}; \
   echo; echo \"=== training exited with code \$? at \$(date) ===\"; \
   exec bash"

echo
echo "started in tmux session '${SESSION}'."
echo
echo "  attach :  tmux attach -t ${SESSION}"
echo "  detach :  Ctrl-b  then  d        (leaves it running)"
echo "  stop   :  attach, then Ctrl-c    (safe: resumes from last.ckpt)"
echo
echo "losses  :  tensorboard --logdir ${RUN_DIR}/tensorboard"
