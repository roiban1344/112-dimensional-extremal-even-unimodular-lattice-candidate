"""S18. Orbit tables of Nebe's lattice L_{56,2}(M): minimal vectors and norm-8 vectors under SL_2(29) and its
centralizer, with completeness checks (needed for gluing A, Table 6, and the reduction of §6.1 (b);
Table D2 [C4], [C6]). Not used in the proof of Theorem A.
S18. Nebe の格子 L_{56,2}(M) の軌道表: SL_2(29) とその中心化群による最小ベクトルとノルム 8 のベクトル。完全性の検査つき（貼り方 A、表 6、
§6.1 (b) の帰着に使う。表 D2 の [C4], [C6]）。定理 A の証明には使わない。

Steps (the same methods as for L; see produce/p1, p2, p3 and verify/v3, v4):
手順（L と同じ方法。produce/p1, p2, p3 と verify/v3, v4 を参照）:
  1. the centralizer C_M of SL_2(29) in Aut(L_{56,2}(M)) (expected order 24);
     Aut(L_{56,2}(M)) における SL_2(29) の中心化群 C_M（期待値は位数 24）;
  2. min L_{56,2}(M) = 6 (by the enumeration that cannot miss vectors, on a BKZ-40 basis) and hence the
     extremal theta series: 15,590,400 minimal vectors, 36,957,286,800 of norm 8;
     min L_{56,2}(M) = 6（BKZ-40 基底での取りこぼしの無い列挙）。よってテータ級数は極値で、最小ベクトル 15,590,400 本、ノルム 8 は 36,957,286,800 本;
  3. orbits of minimal vectors by closure (p2) and their completeness by canonical forms and sizes (v3);
     最小ベクトルの軌道を閉包で（p2）、完全性を標準形と大きさの和で（v3）;
  4. orbits of norm-8 vectors under G* = SL_2(29) C_M by minimal-sum elements (p3) and their completeness:
     the classes x_i u_k give distinct canonical forms whose orbit sizes add up to |L_8| (v4).
     ノルム 8 のベクトルの G* = SL_2(29) C_M-軌道を最小和元で（p3）、完全性: 類 x_i u_k の相異なる標準形の軌道の大きさの和が |L_8|（v4）。
Output in out/nebe/: M_tables.npz (units, min_reps, norm8_reps, K8).
出力（out/nebe/）: M_tables.npz（units, min_reps, norm8_reps, K8）。

Requires SageMath (for steps 1, 2). Run time: about 3 hours (after lattices.py). Memory: about 3 GB.
SageMath が要る（手順 1, 2）。実行時間: 約 3 時間（lattices.py のあと）。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/nebe/tables.py
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))
from r112 import certs, exact, group, mod2  # noqa: E402
from r112 import enumerate as exact_enum  # noqa: E402
from r112.report import Report  # noqa: E402

N6, N8 = 15590400, 36957286800


def main():
    from sage.all import ZZ, matrix
    from lattices import load
    from p1_centralizer import commutant_basis, elements_of_norm
    from p2_minimal_orbits import minimal_orbits
    from p3_norm8_orbits import norm8_orbits
    from p6_reduction import bkz_transform
    ap = argparse.ArgumentParser(description="S18: orbit tables of L_{56,2}(M)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s18_nebe_tables", a.out)
    out = os.path.join(rep.out_dir, "nebe")
    Gs, gens_s = load("M", rep.out_dir)
    G = np.array(Gs.rows(), dtype=np.int64)
    gens = [np.array(g.rows(), dtype=np.int64) for g in gens_s]

    # 1. centralizer / 中心化群
    half = elements_of_norm(commutant_basis(gens_s, 56), Gs, 1)
    rep.check("1. the centralizer of SL_2(29) has order 24", 2 * len(half) == 24 and all(u * Gs * u.transpose() == Gs for u in half))
    U = np.array([np.array(u.rows(), dtype=np.int64) for u in half])

    # 2. minimum 6 / 最小 6
    T = bkz_transform(G, 40)
    assert abs(exact.det_bareiss(T)) == 1
    Gr = exact.i64_matmul(exact.i64_matmul(T, G), T.T)
    found, nodes = exact_enum.short_vectors(Gr, 4, os.path.join(rep.out_dir, "build"))
    rep.value("2. nodes of the enumeration of norm <= 4", nodes)
    rep.check("2. min L_{56,2}(M) = 6 (no nonzero vector of norm <= 4)", len(found) == 0)

    # 3. minimal vectors / 最小ベクトル
    S = group.sl2_29_mod_sign(gens)
    B6 = [t for t in T if int(t @ G @ t) == 6]
    seeds = B6 + [x + s * y for i, x in enumerate(B6) for y in B6[:i] for s in (1, -1) if int((x + s * y) @ G @ (x + s * y)) == 6]
    R6 = minimal_orbits(G, S, seeds, rep)
    can = mod2.Canonical(S)
    f6, sz6 = can.canon_and_orbit_size(mod2.pack(R6))
    rep.check("3. distinct canonical forms, orbit sizes add up to 15590400", len(set(f6.tolist())) == len(R6) and int(2 * sz6.sum()) == N6)
    rep.value("3. orbits of minimal vectors {size: number}", {int(k): int(v) for k, v in zip(*np.unique(2 * sz6, return_counts=True))})

    # 4. norm-8 vectors / ノルム 8 のベクトル
    X, sizes = norm8_orbits(G, S, R6, U, os.path.join(out, "p3_M"), rep)
    X = X.astype(np.int64)
    Y = np.concatenate([exact.i64_matmul(X, u)[:, None, :] for u in U], axis=1).reshape(-1, 56)
    rep.check("4. every x_i u_k has norm 8", (exact.norms(Y, G) == 8).all())
    forms = np.empty(len(Y), dtype=np.uint64)
    osz = np.empty(len(Y), dtype=np.int64)
    for i in range(0, len(Y), 1 << 15):
        forms[i:i + (1 << 15)], osz[i:i + (1 << 15)] = can.canon_and_orbit_size(mod2.pack(Y[i:i + (1 << 15)]))
    K8, first = np.unique(forms, return_index=True)
    rep.value("4. G*-orbits found, distinct G-orbits (canonical forms)", [len(X), len(K8)])
    rep.check("4. the distinct G-orbits have sizes adding up to |L_8|", int(2 * osz[first].sum()) == N8)
    certs.save_npz(os.path.join(out, "M_tables.npz"), units=U, min_reps=R6.astype(np.int16), norm8_reps=X.astype(np.int16), K8=K8)
    rep.finish()


if __name__ == "__main__":
    main()
