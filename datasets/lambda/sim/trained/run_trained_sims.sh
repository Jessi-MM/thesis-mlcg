#!/bin/bash
# Trained-model CG simulations for lambda: earliest and latest checkpoint.
#
# Same convention as bba (sim_ep001.yaml / sim_ep185.yaml): simulate an early and
# a late checkpoint so the FES comparison shows what training actually bought.
# Everything except model_file and filename is identical to the prior-only
# baseline, so any difference in the resulting FES is the network's doing.
#
# ep15 runs first: if anything goes wrong we would rather have the final model.
set -u
B=/storage/mi/itzeljem79/Projects/mlcg/.venv/bin
CFG=/storage/mi/itzeljem79/Projects/thesis-mlcg/datasets/lambda/sim/trained

for TAG in ep15 ep0; do
  OUT=/srv/data/itzeljem79/mlcg-tk/lambda/trained_sims/$TAG/prod
  mkdir -p "$OUT"
  echo "=============================================="
  echo "simulating $TAG   start $(date)"
  echo "=============================================="
  "$B/mlcg-nvt_langevin" --config "$CFG/sim_$TAG.yaml" 2>&1 | tail -3
  echo "$TAG done $(date)"
  echo
done
echo "ALL TRAINED SIMS DONE $(date)"
