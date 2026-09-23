#!/bin/bash
# Wait for step 1a (process_raw_dataset) to finish, check it really succeeded,
# then run step 1b (build_neighborlists). Leaves marker files behind either way.

WORK=/srv/data/itzeljem79/mlcg-tk/bba
BIN=/storage/mi/itzeljem79/Projects/mlcg/.venv/bin
NBATCH=10

cd "$WORK" || exit 1

echo "[$(date)] watcher started; waiting for process_raw_dataset to exit"
while pgrep -f "gen_input_data process_raw_dataset" > /dev/null; do
    sleep 60
done
echo "[$(date)] process_raw_dataset is no longer running"

# Did it actually finish, or did it die partway?
nc=$(ls out/1FME_batch_*_cg_coords.npy 2>/dev/null | wc -l)
nf=$(ls out/1FME_batch_*_cg_forces.npy 2>/dev/null | wc -l)
echo "[$(date)] found $nc coord batches and $nf force batches (expected $NBATCH each)"

if [ "$nc" -ne "$NBATCH" ] || [ "$nf" -ne "$NBATCH" ]; then
    echo "[$(date)] STEP 1a INCOMPLETE — not starting neighbourlists"
    touch STEP1A_INCOMPLETE
    exit 1
fi
touch STEP1A_DONE

echo "[$(date)] starting build_neighborlists"
"$BIN/mlcg-tk-gen_input_data" build_neighborlists \
    --config bba.yaml \
    --config bba_priors_badn_min_pair_4.yaml

if [ $? -eq 0 ]; then
    echo "[$(date)] STEP 1b DONE"
    touch STEP1B_DONE
else
    echo "[$(date)] STEP 1b FAILED"
    touch STEP1B_FAILED
fi
