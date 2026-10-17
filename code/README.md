# Code for “An extremal even unimodular lattice in dimension 112”

[日本語版](README.ja.md)

This directory contains every computation used in the paper. Table D2 of the paper lists them
as [C1]–[C27]; the last section of this file maps each row to the scripts here.

## What is proved, and what has to be trusted

Theorem A (L₁₁₂ has minimum 10) follows from Lemmas 3.2 and 4.2 (proved by hand, and checked in
Lean, `lean/`) and the computational facts F0–F5 of §4.4. The scripts in **`verify/`** check
F0–F5 from two kinds of input:

* the four published files in `data/` (the Gram matrix of L, the generators of SL₂(29), τ, and a
  reduced Gram matrix of L₁₁₂), whose SHA-256 hashes are fixed in `r112/data.py`;
* the **certificate files** in `certificates/` (orbit representatives, bounding representatives,
  basis changes; see `certificates/MANIFEST.md`), whose hashes are fixed in
  `certificates/SHA256SUMS`.

To accept Theorem A you have to trust the verification code (about 1,500 lines of Python and one
C program, using only numpy) and the hand-proved lemmas. You do **not** have to trust the code
that found the certificates (`produce/`): any property of the certificates that the proof uses
is re-checked by `verify/`.

## Layout

| directory | content |
|---|---|
| `r112/` | small shared library (numpy only): reading the data, exactness guards, the group, canonical forms modulo 2, minimal vectors |
| `verify/` | the checks of F0–F5 (v0–v7) and a consistency test (v8) |
| `produce/` | how the certificates were found (p1–p6); not needed to check the proof |
| `crosscheck/` | second implementations of the checks, written separately with Claude without reading `verify/`, `produce/` or `r112/`. They check the same data by different code, so an error in the main implementation would show up as a disagreement: `all_data/` (every certificate and F1, by a different method) and `second_impl/` (canonical forms, design identities, the odd classes) |
| `supplementary/` | computations behind statements of the paper that are not used in the proof of Theorem A (the classification of gluings, the automorphism group, …) |

## Quick start

```bash
bash code/verify/fetch.sh     # download the large certificate files and check their hashes
bash code/verify/quick.sh     # about 30 minutes: everything except the two long steps
bash code/verify/full.sh 3    # about 12 CPU hours: the complete check (3 threads for v7)
```

Run the commands from the repository root. Each step prints `[ok]`/`[FAIL]` for every check and
writes `out/<step>.json` with the results, the software versions and the running time.

## The verification steps

Times were measured on the machine used for the paper (8 cores, 15 GB, shared with other jobs).

| step | paper | what is checked | time | memory |
|---|---|---|---|---|
| `v0_inputs.py` | F0, F4, Lemma 3.2 | L even unimodular; g₁, g₂ preserve G, generate a group of order 24,360 containing −I, commute with τ; τ² = −5, τGτᵀ = 5G; (3.1) even unimodular; the distributed Gram matrix of L₁₁₂ is the same lattice | 20 s | < 1 GB |
| `v1_min_L.py` + `enum_interval.c` | F1, §6.1 (a), Table 9a | min L = 6 by an interval-arithmetic enumeration that cannot miss vectors; positive controls E₈, Leech, Leech ⊥ Leech ⊥ E₈ | 45 min | < 1 GB |
| `v2_theta.py` | §2.3, §2.6, App. B | extremal theta series (\|L₆\|, \|L₈\|), Siegel means, Tables 3, 5, 8, B1, B2 | 1 s | — |
| `v3_minimal_orbits.py` | F2, §6.2 | the 656 orbits cover the 15,590,400 minimal vectors (by expansion and by canonical forms) | 3 min | 2 GB |
| `v4_norm8_orbits.py` | F3, §6.4 | the 1,527,984 vectors x_i u_k give 1,517,130 distinct regular orbits covering \|L₈\| | 10 min | 2 GB |
| `v5_odd_bounding.py` | F5a, Cor. 5.2, Table 11 | 656 odd bounding representatives | 3 min | 2 GB |
| `v6_canonical_lookup.py` | F5b, Prop. 6.3, Table 10 | no class x_i u_k τ + 2L contains a norm-8 vector (canonical forms); also type (6,6) | 10 min | 2 GB |
| `v7_even_bounding.py` | F5b, Cor. 5.3, Table 12 | 1,527,984 even bounding representatives (resumable; `--threads`) | 8 CPU h | 3 GB |
| `v8_design_identities.py` | §6.3 | 18 spherical-design identities (consistency test, not needed for the proof) | 10 min | 2 GB |

F5b is checked twice by unrelated arguments (v6 and v7); either one suffices for the proof.

### How exactness is ensured

All decisions are made in integer arithmetic. Where a matrix product of integer matrices is
computed in floating point (for speed), the code first asserts from the actual inputs that every
partial sum is an integer below 2²⁴ (float32) or 2⁵³ (float64); such products are exact. The
helpers are in `r112/exact.py`; each script's header states which products it computes this way.
The enumeration of F1 uses outward-rounded interval arithmetic (`verify/enum_interval.c`, header).

## Requirements

* `verify/`, `crosscheck/second_impl/`: Python ≥ 3.10 with numpy (tested with Python 3.12, numpy 2.5),
  a C compiler with 128-bit integers (gcc or clang) for v1. No other package.
* `produce/`, `supplementary/`, `crosscheck/all_data/`: SageMath 10 (which includes PARI/GP, GAP and
  fpylll); tested with SageMath 10.9. `environment.yml` at the repository root installs it with
  conda/micromamba.
* `lean/`: Lean 4 and Mathlib (versions pinned in `lean/lean-toolchain` and `lean/lake-manifest.json`).

## Producing the certificates

| script | produces | time |
|---|---|---|
| `produce/p1_centralizer.py` | M′ and the centralizer C (`centralizer.npz`, `commutant.npz`) | 10 min |
| `produce/p2_minimal_orbits.py` | `min_reps.npz` (closure, §6.2) | 10 min |
| `produce/p3_norm8_orbits.py` | `norm8_reps.npz` (minimal-sum elements, §6.4) | 1.5 h |
| `produce/p4_odd_descent.py` | `odd_bounding.npz` (descent, §6.6) | 3 min |
| `produce/p5_even_descent.py` | `even_bounding.npz` (descent, §6.7; 6 parts, resumable) | ≈ 190 CPU h |
| `produce/p6_reduction.py` | `reduction_L.txt`, `reduction_R.txt` | 1 min |

Each producer compares its output with the distributed certificate. p4 and p5 reproduce the
certificates bit for bit; p2 and p3 may choose other representatives, so they are compared by
the set of canonical forms (the same orbits).

## Correspondence with Table D2 of the paper

| Table D2 | scripts |
|---|---|
| [C1] | verify/v2_theta.py |
| [C2] | verify/v0_inputs.py, supplementary/gluings.py (d), produce/p1_centralizer.py |
| [C3] | supplementary/gluings.py, supplementary/nebe/gluings_m.py (a), (b) |
| [C4] | supplementary/gluing_b.py, supplementary/nebe/gluings_m.py (c), (d) |
| [C5] | verify/v1_min_L.py + verify/enum_interval.c, produce/p6_reduction.py |
| [C6] | supplementary/nebe/reduction_to_nebe.py (with nebe/lattices.py, nebe/tables.py) |
| [C7] | supplementary/float_enumerations.py svp, qfminim |
| [C8] | produce/p2_minimal_orbits.py, verify/v3_minimal_orbits.py, verify/v8_design_identities.py |
| [C9] | produce/p3_norm8_orbits.py, verify/v4_norm8_orbits.py, verify/v8_design_identities.py |
| [C10] | produce/p4_odd_descent.py, verify/v5_odd_bounding.py, crosscheck/second_impl/odd_classes.py |
| [C11] | verify/v6_canonical_lookup.py, crosscheck/second_impl/even_classes.py |
| [C12] | supplementary/float_enumerations.py cvp610, cvp88 |
| [C13] | produce/p5_even_descent.py, verify/v7_even_bounding.py |
| [C14] | supplementary/frame_stabilizer.py |
| [C15] | supplementary/sample_check.py |
| [C16] | supplementary/node_estimate.py |
| [C17] | supplementary/frames.py |
| [C18] | supplementary/fixed_sublattices.py |
| [C19] | supplementary/anatomy_b.py |
| [C20, C21] | (not yet included / 未収録) |
| [C22] | lean/ (EEUL/R112/Chain.lean) |
| [C23] | supplementary/nebe/identify_nebe.py (with nebe/lattices.py, nebe/cycloquat.py) |
| [C24] | supplementary/automorphisms/aut_normalizer.py |
| [C25] | supplementary/automorphisms/group_facts.g, out_facts.g, aut_diagonal.py |
| [C26] | supplementary/automorphisms/aut_structure.py, aut_character.py, endomorphisms.py |
| [C27] | crosscheck/all_data/check_all.py, min_L.py, min_L_enum.c |
