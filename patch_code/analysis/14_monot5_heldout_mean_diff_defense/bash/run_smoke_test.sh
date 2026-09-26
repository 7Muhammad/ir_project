#!/usr/bin/env bash
# bash/run_smoke_test.sh — full pipeline on tiny data (configs/smoke.yaml),
# CPU/login-feasible (also runs fine on GPU if available), writes to
# outputs_smoke/ so it never touches the real outputs/ directory. Verifies
# every stage runs end-to-end: direction fitting, all 5 encoder position
# masks, scale selection, encoder+decoder interventions, clean-damage
# evaluation, IID test, all 3 attack-OOD protocols, aggregation, plots.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_smoke_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Experiment 14 SMOKE TEST ==="
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" --force

echo ""
echo "=== Smoke test complete. Spot-checking key outputs exist: ==="
for f in \
    "outputs_smoke/candidate_heads.json" \
    "outputs_smoke/split_manifest.json" \
    "outputs_smoke/directions/iid/directions.pt" \
    "outputs_smoke/scale_selection/selected_scales.json" \
    "outputs_smoke/test/iid_test_per_example.csv" \
    "outputs_smoke/test/clean_damage_per_example.csv" \
    "outputs_smoke/ood_test_per_example.csv" \
    "outputs_smoke/ood_clean_damage_per_example.csv" \
    "outputs_smoke/full_recovery_summary.csv" \
    "outputs_smoke/plots/figure_a_recovery_vs_damage.png" \
    "outputs_smoke/plots/figure_b_encoder_conditions.png" \
    "outputs_smoke/plots/figure_c_ood_generalization.png" \
    ; do
    if [[ -s "${EXP_DIR}/${f}" ]]; then
        echo "  OK   ${f}"
    else
        echo "  MISSING/EMPTY  ${f}"
        exit 1
    fi
done
echo "=== All expected smoke outputs present. ==="
