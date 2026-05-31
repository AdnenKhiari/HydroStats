#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

VERSIONS=("Baseline V0" "V1" "V2" "V1+2")
STEP1_MODEL="${STEP1_MODEL:-gemini-2.5-flash}"
STEP2_MODEL="${STEP2_MODEL:-gemini-2.5-flash}"
STEP3_MODEL="${STEP3_MODEL:-gemini-2.5-flash}"
STEP1_EXTRA_ARGS="${STEP1_EXTRA_ARGS:-}"
STEP2_EXTRA_ARGS="${STEP2_EXTRA_ARGS:-}"
STEP3_EXTRA_ARGS="${STEP3_EXTRA_ARGS:-}"

run_step1() {
  echo
  echo "============================================================"
  echo "Step 1: Thematic analysis on all versions"
  echo "============================================================"

  for version in "${VERSIONS[@]}"; do
    echo
    echo "[Step 1] Version: $version"
    python run_thematic.py --versions "$version" --model "$STEP1_MODEL" ${STEP1_EXTRA_ARGS}
  done
}

run_step2() {
  echo
  echo "============================================================"
  echo "Step 2: Codification on all versions"
  echo "============================================================"

  for version in "${VERSIONS[@]}"; do
    echo
    echo "[Step 2] Version: $version"
    python run_codification.py --versions "$version" --model "$STEP2_MODEL" ${STEP2_EXTRA_ARGS}
  done
}

run_step3() {
  echo
  echo "============================================================"
  echo "Step 3: Theme generation on all versions"
  echo "============================================================"

  for version in "${VERSIONS[@]}"; do
    echo
    echo "[Step 3] Version: $version"
    python run_theme_generation.py --versions "$version" --model "$STEP3_MODEL" ${STEP3_EXTRA_ARGS}
  done
}

main() {
  run_step1
  run_step2
  run_step3

  echo
  echo "============================================================"
  echo "All steps completed successfully"
  echo "============================================================"
}

main "$@"
