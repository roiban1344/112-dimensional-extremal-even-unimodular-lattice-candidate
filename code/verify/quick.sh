#!/usr/bin/env bash
# Quick verification (about 30 minutes, under 4 GB of memory): every step of the proof of
# Theorem A except the long ones, which are replaced by their fast counterparts:
#   F1 by the interval enumeration is run on the positive controls only (v1 --controls-only),
#   F5b is decided by canonical forms (v6) instead of the even bounding representatives (v7).
# 手早い検算（約 30 分、メモリ 4 GB 未満）。定理 A の証明の各段のうち、長いものを速い方法に置き換えて走らせる:
#   F1 の区間演算の列挙は陽性対照だけ（v1 --controls-only）、F5b は偶の下界代表元（v7）の代わりに標準形（v6）で判定する。
# Run from the repository root / リポジトリの根から:  bash code/verify/quick.sh
set -euo pipefail
PY=${PYTHON:-python3}
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$PY "$D/v0_inputs.py"
$PY "$D/v1_min_L.py" --controls-only
$PY "$D/v2_theta.py"
$PY "$D/v3_minimal_orbits.py"
$PY "$D/v4_norm8_orbits.py"
$PY "$D/v5_odd_bounding.py"
$PY "$D/v6_canonical_lookup.py"
echo "quick verification passed (F1 itself and v7 are in full.sh) / 手早い検算は通過（F1 そのものと v7 は full.sh）"
