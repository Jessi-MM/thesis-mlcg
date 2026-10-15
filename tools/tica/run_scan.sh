#!/bin/bash
# Lag scan for all three featurisations, on a subsample of segments.
# Sequential on purpose: the covariance update is already BLAS-multithreaded
# (measured ~24 GFLOPS), so running the three in parallel would thrash.
set -u
P=/storage/mi/itzeljem79/Projects/mlcg/.venv/bin/python3
T=/storage/mi/itzeljem79/Projects/thesis-mlcg/tools/tica
CACHE=/srv/data/itzeljem79/mlcg-tk/lambda/tica_cache
TOP=/srv/data/itzeljem79/mlcg-tk/lambda/out/lambda_cg_structure.pdb
SEGS=3000
LAGS="20 50 100 160 250"

cd "$T" || exit 1
echo "=============================================="
echo "lag scan started $(date)"
echo "segments subsampled : $SEGS of 15700"
echo "lags (frames)       : $LAGS"
echo "cis split           : --min-bond-nm 0.34 (matches the training-data filter)"
echo "=============================================="

for F in dihedrals distances both; do
  echo
  echo "##############  featurisation: $F  ##############"
  echo "start $(date)"
  "$P" fit_reference_tica.py \
      --cache "$CACHE" --topology "$TOP" \
      --features "$F" --lags $LAGS \
      --max-segments "$SEGS" --min-bond-nm 0.34 \
      2>&1 | grep -vE "^  \[(train|test)\] [0-9,]+ blocks$"
  echo "end   $(date)"
done

echo
echo "=============================================="
echo "lag scan finished $(date)"
echo "=============================================="
