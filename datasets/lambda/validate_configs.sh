#!/bin/bash
# Parse-only validation of the lambda pipeline configs (runs nothing, writes nothing).
cd /storage/mi/itzeljem79/Projects/thesis-mlcg/datasets/lambda || exit 1
B=/storage/mi/itzeljem79/Projects/mlcg/.venv/bin
check () {
  desc="$1"; shift
  out=$("$@" --help 2>&1); rc=$?
  errs=$(printf "%s" "$out" | grep -icE "error|invalid|unexpected|traceback")
  printf "%-20s exit=%-3s error_lines=%s\n" "$desc" "$rc" "$errs"
  if [ "$rc" -ne 0 ] || [ "$errs" -gt 0 ]; then
      printf "%s\n" "$out" | grep -iE "error|invalid|unexpected|traceback" | head -5
  fi
}
echo "started $(date)"
check "1b neighborlists" $B/mlcg-tk-gen_input_data build_neighborlists --config lambda.yaml --config lambda_priors.yaml
check "2 statistics"     $B/mlcg-tk-fit_priors compute_statistics --config lambda_stats.yaml --config lambda_priors.yaml
check "3 fit_priors"     $B/mlcg-tk-fit_priors fit_priors --config lambda_fit.yaml
check "4 delta_forces"   $B/mlcg-tk-produce_delta_forces produce_delta_forces --config lambda_delta_forces.yaml
check "5 packaging"      $B/mlcg-tk-package_training_data package_training_data --config lambda_packaging.yaml
echo "ALL CHECKS DONE $(date)"
