"""S14. Fixed sublattices of elements of order 3, 5, 7 of SL_2(29) in R, and BKZ on R (paper §7.3;
Table D2 [C18]). These are consistency tests that are NOT evidence for extremality (they cannot tell
gluing B from gluing C); they are not used in the proof of Theorem A.
S14. R における SL_2(29) の位数 3, 5, 7 の元の固定部分格子と、R への BKZ（論文 §7.3。表 D2 の [C18]）。極値性の証拠に**ならない**整合性の検査
（貼り方 B と C を区別できない）。定理 A の証明には使わない。

For an element h of prime order p, the fixed sublattice R^h = { z in R : z diag(h, h) = z } is computed
exactly (kernel over Q, saturated in Z^112). Its minimum m is established exactly: a vector of norm m is
exhibited (found by PARI, its norm checked in integers), and the enumeration that cannot miss vectors
(verify/enum_interval.c) finds no nonzero vector of norm <= m - 2. Expected (paper): dimensions 32, 16,
16, minima 12, 10, 14, determinants 3^32, 5^16, 7^16. The order-5 section contains vectors of norm 10, so
min R <= 10.
素数位数 p の元 h について、固定部分格子 R^h を厳密に求める（Q 上の核、Z^112 で飽和）。最小 m は厳密に決める: ノルム m のベクトルを示し
（PARI で見つけ、ノルムを整数で確かめる）、取りこぼしの無い列挙（verify/enum_interval.c）でノルム m - 2 以下の 0 でないベクトルが無いことを確かめる。
期待値（論文）: 次元 32, 16, 16、最小 12, 10, 14、行列式 3^32, 5^16, 7^16。位数 5 の固定部分格子はノルム 10 のベクトルを含むので min R ≤ 10。

BKZ-30 and BKZ-40 on R itself: the shortest vector found (an upper bound only; floating point).
R そのものへの BKZ-30 と BKZ-40: 見つかった最短（上界にすぎない。浮動小数点）。

Requires SageMath (PARI, fpylll). Run time: about 20 minutes. Memory: about 1 GB.
SageMath（PARI, fpylll）が要る。実行時間: 約 20 分。メモリ: 約 1 GB。
Run / 実行:  sage -python code/supplementary/fixed_sublattices.py [--tau B]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from r112 import certs, data  # noqa: E402
from r112 import enumerate as exact_enum  # noqa: E402
from r112.report import Report  # noqa: E402

EXPECTED = {3: (32, 12), 5: (16, 10), 7: (16, 14)}


def main():
    from fpylll import BKZ, GSO, LLL, IntegerMatrix
    from fpylll.algorithms.bkz2 import BKZReduction
    from sage.all import QQ, ZZ, block_matrix, identity_matrix, matrix, pari
    ap = argparse.ArgumentParser(description="S14: fixed sublattices and BKZ")
    ap.add_argument("--tau", choices=("C", "B"), default="C")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s14_fixed_sublattices" + ("_B" if a.tau == "B" else ""), a.out)
    n = 56
    G = matrix(ZZ, data.gram(a.data).tolist())
    g1, g2 = [matrix(ZZ, g.tolist()) for g in data.generators(a.data)]
    X = matrix(ZZ, (data.tau(a.data) if a.tau == "C" else certs.load("negative_control_B.npz", "tau_class_B", a.certificates)).tolist())
    I = identity_matrix(ZZ, n)
    B = block_matrix(ZZ, [[I, X], [0 * I, 2 * I]], subdivide=False)
    GR = (B * block_matrix(ZZ, [[G, 0], [0, G]], subdivide=False) * B.transpose() / 2).change_ring(ZZ)
    rep.check("Gram matrix of R: even, det 1", all(GR[i, i] % 2 == 0 for i in range(112)) and GR.det() == 1)
    Binv = B.change_ring(QQ).inverse()
    # Elements of order 3, 5, 7 as powers of words in g1, g2 (g1 has order 5). / g1, g2 の語の冪として位数 3, 5, 7 の元。
    elts = {5: g1}
    for word in (g2, g1 * g2, g1 * g1 * g2, g1 * g2 * g2, g1 * g2 * g1 * g2 * g2):
        o = word.multiplicative_order()
        for p in (3, 7):
            if o % p == 0 and p not in elts:
                elts[p] = word ** (o // p)
    rep.check("elements of order 3, 5, 7 found", sorted(elts) == [3, 5, 7])
    build = os.path.join(rep.out_dir, "build")
    for p, h in sorted(elts.items()):
        assert h ** p == I and h != I
        hh = block_matrix(ZZ, [[h, 0 * h], [0 * h, h]], subdivide=False)
        Kc = (B * hh * Binv - 1).change_ring(QQ).left_kernel().basis_matrix()       # R-coordinates fixed by h
        Kc = (Kc * Kc.denominator()).change_ring(ZZ).saturation()
        Gs = Kc * GR * Kc.transpose()
        dim, det = Kc.nrows(), Gs.det()
        U = pari(Gs).qflllgram().sage().transpose()
        assert abs(U.det()) == 1
        Gr = U * Gs * U.transpose()
        res = pari(Gr).qfminim(None, 1, 0)                         # PARI: the minimum and one minimal vector
        m = int(res[1])
        v = matrix(ZZ, [list(res[2].sage().column(0))])
        assert int((v * Gr * v.transpose())[0, 0]) == m
        found, nodes = exact_enum.short_vectors([[int(c) for c in r] for r in Gr.rows()], m - 2, build)
        rep.value(f"order {p}: dimension, determinant, minimum, nodes of the exact enumeration below the minimum",
                  [dim, str(det.factor()), m, nodes])
        rep.check(f"order {p}: exact minimum {m} (a vector of norm {m}; none of norm <= {m - 2})", len(found) == 0)
        if a.tau == "C":
            rep.check(f"order {p}: dimension and minimum as in the paper {EXPECTED[p]}", (dim, m) == EXPECTED[p])
    for bs in (30, 40):
        A = IntegerMatrix.from_matrix([[int(c) for c in r] for r in GR.rows()])
        M = GSO.Mat(A, gram=True, float_type="mpfr", flags=GSO.INT_GRAM)
        M.update_gso()
        LLL.Reduction(M)()
        BKZReduction(M)(BKZ.Param(block_size=bs, max_loops=4, strategies=BKZ.DEFAULT_STRATEGY))
        rep.value(f"BKZ-{bs} on R: shortest norm in the reduced basis (upper bound only)", int(min(A[i, i] for i in range(112))))
    rep.finish()


if __name__ == "__main__":
    main()
