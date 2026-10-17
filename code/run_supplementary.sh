#!/usr/bin/env bash
# Run the supplementary computations one after another (each writes out/<step>.json and a log
# out/logs/<name>.log); a failing step does not stop the others. Requires SageMath and GAP.
# 補足の計算を順に走らせる（各段は out/<step>.json とログ out/logs/<名前>.log を書く）。失敗した段があっても残りは続ける。SageMath と GAP が要る。
# Run from the repository root / リポジトリの根から:  bash code/run_supplementary.sh [names...]
set -u
PY=${PYTHON:-python3}
C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$C/../out/logs"; mkdir -p "$LOG"
declare -A STEP=(
  [gluings]="$PY $C/supplementary/gluings.py"
  [frame_stabilizer]="$PY $C/supplementary/frame_stabilizer.py"
  [node_estimate]="$PY $C/supplementary/node_estimate.py"
  [sample_check]="$PY $C/supplementary/sample_check.py"
  [frames]="$PY $C/supplementary/frames.py"
  [gluing_b]="$PY $C/supplementary/gluing_b.py"
  [anatomy_b]="$PY $C/supplementary/anatomy_b.py"
  [fixed_sublattices]="$PY $C/supplementary/fixed_sublattices.py"
  [float_svp]="$PY $C/supplementary/float_enumerations.py svp"
  [float_qfminim]="$PY $C/supplementary/float_enumerations.py qfminim"
  [float_cvp610]="$PY $C/supplementary/float_enumerations.py cvp610"
  [aut_normalizer]="$PY $C/supplementary/automorphisms/aut_normalizer.py"
  [aut_diagonal]="$PY $C/supplementary/automorphisms/aut_diagonal.py"
  [aut_structure]="$PY $C/supplementary/automorphisms/aut_structure.py"
  [aut_character]="$PY $C/supplementary/automorphisms/aut_character.py"
  [endomorphisms]="$PY $C/supplementary/automorphisms/endomorphisms.py"
  [group_facts]="gap -q $C/supplementary/automorphisms/group_facts.g $C/supplementary/automorphisms/out_facts.g"
  [nebe_lattices]="$PY $C/supplementary/nebe/lattices.py"
  [identify_nebe]="$PY $C/supplementary/nebe/identify_nebe.py"
  [nebe_tables]="$PY $C/supplementary/nebe/tables.py"
  [nebe_gluings]="$PY $C/supplementary/nebe/gluings_m.py"
  [reduction_to_nebe]="$PY $C/supplementary/nebe/reduction_to_nebe.py"
)
ORDER=(gluings frame_stabilizer node_estimate sample_check frames gluing_b anatomy_b fixed_sublattices float_svp float_qfminim
       float_cvp610 aut_normalizer aut_diagonal aut_structure aut_character endomorphisms group_facts nebe_lattices identify_nebe
       nebe_tables nebe_gluings reduction_to_nebe)
[ $# -gt 0 ] && ORDER=("$@")
for s in "${ORDER[@]}"; do
  echo "=== $s ($(date '+%F %T'))"
  if ${STEP[$s]} > "$LOG/$s.log" 2>&1; then echo "    ok"; else echo "    FAILED (see out/logs/$s.log)"; fi
done
