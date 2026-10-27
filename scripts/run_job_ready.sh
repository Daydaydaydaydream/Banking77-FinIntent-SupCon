#!/usr/bin/env bash
# Run from any directory; activate the documented Python environment first.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/banking77-matplotlib"
ROOT_OUT="${BANKING77_OUTPUT_DIR:-$PWD/outputs/job_ready_gpu_v2}"
case "${1:-}" in
  prepare)
    python -m data.prepare_data
    python -m prepare_runtime --device cuda --download-model
    python -m pip freeze > outputs/environment-wsl.txt
    ;;
  smoke)
    python -m prepare_runtime --device cuda
    python -m run_controlled --protocol configs/controlled_gpu_v2.json --output-dir "$ROOT_OUT" --device cuda --smoke "${@:2}"
    ;;
  train)
    python -m prepare_runtime --device cuda
    python -m run_controlled --protocol configs/controlled_gpu_v2.json --output-dir "$ROOT_OUT" --device cuda --setting 5shot --seed 42 43 44 "${@:2}"
    python -m analysis.controlled_report --root "$ROOT_OUT"
    ;;
  report)
    python -m analysis.controlled_report --root "$ROOT_OUT"
    ;;
  demo)
    python -m predict --selection "$ROOT_OUT/delivery/selected_model.json" \
      --text "Where is my card?" --text "Why do I have to verify my identity?" "${@:2}"
    ;;
  *)
    echo 'Usage: bash scripts/run_job_ready.sh {prepare|smoke|train|report|demo} [extra arguments]'
    exit 2
    ;;
esac
