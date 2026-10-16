#!/usr/bin/env bash
# JGP on the storm-surge splits: choose the neighborhood size on a training-only validation
# set of the first smaller split, then fit every split with that size.
#   STORM_HOLDOUT=random|grouped bash storm_surge/run_jgp_all.sh
# Requires the Python environment in vendor/jumpgp_requirements.txt (PYTHON_BIN, default python3).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
python="${PYTHON_BIN:-python3}"
cores="${CORES:-16}"
holdout="${STORM_HOLDOUT:-random}"
out="${STORM_OUTPUT:-${here}/../results/storm_surge/${holdout}}/jgp"
tmp="${out}/tmp"
mkdir -p "${out}/tuning" "${tmp}"

Rscript "${here}/export_jgp_data.R" n10000 1 "${tmp}/pilot_train.csv.gz" "${tmp}/pilot_test.csv.gz" pilot
for neighbors in ${JGP_NEIGHBOR_GRID:-25 50 100 200 285}; do
  mkdir -p "${out}/tuning/m${neighbors}"
  "${python}" "${here}/run_jgp.py" --experiment pilot --rep 1 \
    --train "${tmp}/pilot_train.csv.gz" --test "${tmp}/pilot_test.csv.gz" \
    --output-dir "${out}/tuning/m${neighbors}" --cores "${cores}" --boundary linear \
    --neighbors "${neighbors}" --mode CEM --draws 300 --prediction-seed 97001
done
neighbors="$("${python}" "${here}/select_jgp_tuning.py" "${out}/tuning")"
echo "Selected JGP neighborhood: ${neighbors}"

for version in n10000 full; do
  for rep in $(seq 1 10); do
    tag="$(printf 'rep%03d' "${rep}")"
    Rscript "${here}/export_jgp_data.R" "${version}" "${rep}" \
      "${tmp}/${version}_${tag}_train.csv.gz" "${tmp}/${version}_${tag}_test.csv.gz"
    mkdir -p "${out}/${version}/${tag}"
    "${python}" "${here}/run_jgp.py" --experiment "${version}" --rep "${rep}" \
      --train "${tmp}/${version}_${tag}_train.csv.gz" --test "${tmp}/${version}_${tag}_test.csv.gz" \
      --output-dir "${out}/${version}/${tag}" --cores "${cores}" --boundary linear \
      --neighbors "${neighbors}" --mode CEM --draws 300 --prediction-seed "$((98000 + rep))"
    rm -f "${tmp}/${version}_${tag}_train.csv.gz" "${tmp}/${version}_${tag}_test.csv.gz"
  done
done
