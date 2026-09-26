#!/usr/bin/env bash
# bash/find_data.sh
# =================
# Stage: data discovery — find DL19 / BM25 / MS MARCO files on disk.
#
# Usage:
#   bash bash/find_data.sh [config_path]
#
# Defaults to: configs/default.yaml
#
# Writes a candidate file report to:
#   outputs/data_discovery/data_candidates.txt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

echo "============================================================"
echo "  Stage: find_data"
echo "  Config: ${CONFIG}"
echo "============================================================"

mkdir -p "${PROJECT_ROOT}/outputs/data_discovery"

python3 - <<PYEOF
import pathlib, sys
sys.path.insert(0, "${PROJECT_ROOT}")
import yaml
from src.find_data import discover_data

cfg_path = pathlib.Path("${CONFIG}")
with open(cfg_path) as fh:
    cfg = yaml.safe_load(fh)

data_cfg = cfg["data"]
report_path = pathlib.Path("${PROJECT_ROOT}") / cfg["outputs"]["base_dir"] / "data_discovery" / "data_candidates.txt"

discover_data(
    upstream_repo_path=data_cfg.get("upstream_repo_path"),
    search_roots=data_cfg.get("search_roots", []),
    report_path=report_path,
    script_dir=pathlib.Path("${PROJECT_ROOT}"),
)
PYEOF

echo ""
echo "Discovery report: ${PROJECT_ROOT}/outputs/data_discovery/data_candidates.txt"
echo ""
echo "Next step:"
echo "  Review the report and set the appropriate paths in ${CONFIG},"
echo "  then run:  bash bash/run_prepare_pairs.sh ${CONFIG}"
