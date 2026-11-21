#!/usr/bin/env bash
# Runs msckf_run headless on the reference sequences and prints a markdown table of accuracy, consistency and speed.
#
#   pixi run benchmark                          # all sequences
#   pixi run benchmark --loop_closure           # extra options are forwarded to msckf_run
#   ONLY=V1_ pixi run benchmark                 # sequences whose name contains V1_
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/build/cpp/msckf_run"
SEQUENCES=("euroc V1_01_easy" "euroc V1_02_medium" "euroc V1_03_difficult" "tartanair P001 --noise_level mid")

# Number following a label in the output of msckf_run, empty if the label is missing
field() { { grep -oE "$1 *[0-9.]+" <<< "$2" || true; } | head -1 | grep -oE "[0-9.]+$" || true; }

echo "| Sequence | Distance | ATE RMSE | Loop-corrected ATE | Orient. RMSE | Inside 3σ | NEES pos / orient | Camera loop | Real-time factor | Wall time |"
echo "|---|---|---|---|---|---|---|---|---|---|"
for entry in "${SEQUENCES[@]}"; do
    read -r source sequence rest <<< "$entry"
    if [[ -n "${ONLY:-}" && "$sequence" != *"$ONLY"* ]]; then continue; fi
    start=$(date +%s)
    # shellcheck disable=SC2086
    output=$(cd "$ROOT" && "$RUN" --source "$source" --sequence "$sequence" --no_viewer $rest "$@" 2>&1) || { echo "| $sequence | failed: $(tail -1 <<< "$output") |"; continue; }
    wall=$(( $(date +%s) - start ))
    distance=$(field "steps," "$output")
    ate=$(field "  - Position RMSE:" "$output")
    corrected=$(field "Loop-corrected position RMSE:" "$output")
    orientation=$(field "Orient. RMSE:" "$output")
    inside=$(field "Inside 3σ:" "$output")
    nees="$(field "NEES: +position" "$output") / $(field ", orientation" "$output")"
    camera=$({ grep -E "With Camera:" <<< "$output" || true; } | sed -E 's/.* - ([0-9.]+) Hz/\1/' | awk '{printf "%.1f", $1}')
    realtime=$(field "Real-time factor:" "$output")
    echo "| ${sequence} | ${distance} m | ${ate} m | ${corrected:-–}${corrected:+ m} | ${orientation}° | ${inside} % | ${nees} | ${camera} Hz | ${realtime} | ${wall} s |"
done
