#!/usr/bin/env bash
# Full verification of Theorem A (about 12 CPU hours; under 4 GB of memory):
#   v0 F0, F4 | v1 F1 (interval enumeration, ~45 min) | v2 theta series | v3 F2 | v4 F3 |
#   v5 F5a | v6 F5b by canonical forms | v7 F5b by even bounding representatives (~8 CPU hours) |
#   v8 design identities (consistency test, not needed for the proof).
# Each step writes out/<step>.json; v7 can be interrupted and resumed.
# 定理 A の全部の検算（約 12 CPU 時間、メモリ 4 GB 未満）。各段は out/<step>.json を書く。v7 は中断して再開できる。
# Run from the repository root / リポジトリの根から:  bash code/verify/full.sh [threads for v7]
set -euo pipefail
PY=${PYTHON:-python3}
T=${1:-1}
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$PY "$D/v0_inputs.py"
$PY "$D/v1_min_L.py"
$PY "$D/v2_theta.py"
$PY "$D/v3_minimal_orbits.py"
$PY "$D/v4_norm8_orbits.py"
$PY "$D/v5_odd_bounding.py"
$PY "$D/v6_canonical_lookup.py"
$PY "$D/v7_even_bounding.py" --threads "$T"
$PY "$D/v8_design_identities.py"
echo "full verification passed / 全部の検算は通過"
