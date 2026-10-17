"""V0. Basic facts F0 and F4 about the input data (paper §3.1, §4.4, Lemma 3.2).
V0. 入力データについての基本的な事実 F0 と F4（論文 §3.1、§4.4、補題 3.2）。

Checks / 確かめること:
  L    G is symmetric with even diagonal, det G = 1, and G is positive definite:
       L is an even unimodular lattice of dimension 56.
       G は対称で対角が偶数、det G = 1、正定値。つまり L は 56 次元の偶ユニモジュラー格子。
  F4   g1, g2 preserve G (g G g^T = G); they generate a group of order 24,360 that contains -I;
       g1, g2 commute with tau; tau^2 = -5 I; tau G tau^T = 5 G; and tau G is antisymmetric
       (tau* = -tau, Appendix C.1).
       g1, g2 は G を保つ。生成する群の位数は 24,360 で -I を含む。g1, g2 は tau と可換。tau^2 = -5I、tau G tau^T = 5G、
       tau G は反対称（tau* = -tau、付録 C.1）。
  F0   R = {(x, x tau + 2y)/sqrt2}: with B = [[I, tau], [0, 2I]], the matrix B diag(G, G) B^T / 2
       is integral and equals (3.1) = [[3G, tau G], [(tau G)^T, 2G]]; it is even, has
       determinant 1 and is positive definite (Lemma 3.2).
       B diag(G, G) B^T / 2 は整数で (3.1) に等しく、偶・行列式 1・正定値（補題 3.2）。
  data certificates/reduction_R.txt is an integer matrix U with det U = +-1 and
       U (3.1) U^T = data/R112_gram_LLL.txt, so the distributed Gram matrix describes the same lattice.
       certificates/reduction_R.txt は det U = ±1 の整数行列 U で U (3.1) U^T = data/R112_gram_LLL.txt。
       配布した Gram 行列は同じ格子を表す。
  sym  the diagonal action of g1, g2 preserves R (so SL_2(29) acts on R; not needed for Theorem A).
       g1, g2 の対角作用は R を保つ（SL_2(29) が R に作用する。定理 A には不要）。

Exactness: integer arithmetic only (numpy int64 with overflow guards, Python integers for
determinants and the rational LDL decomposition).
厳密さ: 整数演算のみ（オーバーフローを確かめた numpy の int64、行列式と有理数の LDL 分解には Python の整数）。

Run time: about 1 minute. Memory: below 1 GB.
実行時間: 約 1 分。メモリ: 1 GB 未満。

Run / 実行:  python3 code/verify/v0_inputs.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group  # noqa: E402
from r112.report import Report  # noqa: E402


def is_even_symmetric(M):
    return (M == M.T).all() and (np.diag(M) % 2 == 0).all()


def main():
    ap = argparse.ArgumentParser(description="V0: basic facts F0, F4")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v0_inputs", a.out)

    # The readers check the SHA-256 hashes of the four files.
    # 読み込みの際に 4 つのファイルの SHA-256 を照合する。
    G = data.gram(a.data)
    g1, g2 = data.generators(a.data)
    t = data.tau(a.data)
    I = np.eye(56, dtype=np.int64)
    mm = exact.i64_matmul
    rep.check("input files match the published SHA-256 hashes", True)

    # --- L ---
    rep.check("G is symmetric with even diagonal", is_even_symmetric(G))
    rep.check("det G = 1", exact.det_bareiss(G) == 1)
    rep.check("G is positive definite (exact rational LDL decomposition)", exact.is_positive_definite(G))

    # --- F4 ---
    for name, g in (("g1", g1), ("g2", g2)):
        rep.check(f"{name} G {name}^T = G", (mm(mm(g, G), g.T) == G).all())
        rep.check(f"{name} tau = tau {name}", (mm(g, t) == mm(t, g)).all())
    rep.check("tau^2 = -5 I", (mm(t, t) == -5 * I).all())
    rep.check("tau G tau^T = 5 G", (mm(mm(t, G), t.T) == 5 * G).all())
    rep.check("tau G is antisymmetric (tau* = -tau)", (mm(t, G) == -mm(t, G).T).all())
    E = group.enumerate_group([g1, g2])
    rep.value("order of <g1, g2>", len(E))
    rep.check("|<g1, g2>| = 24360", len(E) == group.ORDER)
    rep.check("-I is in <g1, g2>", any((e == -I).all() for e in E))
    del E

    # --- F0 (Lemma 3.2) ---
    Z = np.zeros((56, 56), dtype=np.int64)
    B = np.block([[I, t], [Z, 2 * I]])
    B2 = mm(mm(B, np.block([[G, Z], [Z, G]])), B.T)
    rep.check("B diag(G, G) B^T has even entries", (B2 % 2 == 0).all())
    GR = B2 // 2
    rep.check("B diag(G, G) B^T / 2 equals (3.1) = [[3G, tau G], [(tau G)^T, 2G]]", (GR == data.gram_R(G, t)).all())
    rep.check("(3.1) is symmetric with even diagonal", is_even_symmetric(GR))
    rep.check("det (3.1) = 1", exact.det_bareiss(GR) == 1)
    rep.check("(3.1) is positive definite (exact rational LDL decomposition)", exact.is_positive_definite(GR))

    # --- the distributed reduced Gram matrix ---
    U = certs.load_matrix("reduction_R.txt", 112, a.certificates)
    rep.check("reduction_R.txt: det U = +-1", abs(exact.det_bareiss(U)) == 1)
    rep.check("U (3.1) U^T = data/R112_gram_LLL.txt", (mm(mm(U, GR), U.T) == data.gram_R_reduced(a.data)).all())

    # --- the diagonal action on R (rows of B are a basis of sqrt2 R in L + L) ---
    # g acts on R iff M = B diag(g, g) B^{-1} is integral; then M (3.1) M^T = (3.1) says it is an isometry.
    # B^{-1} = [[I, -tau/2], [0, I/2]], so 2 B^{-1} = [[2I, -tau], [0, I]] is integral.
    # g が R に作用する ⟺ M = B diag(g, g) B^{-1} が整数。さらに M (3.1) M^T = (3.1) なら等長変換。
    # B^{-1} = [[I, -tau/2], [0, I/2]] なので 2B^{-1} = [[2I, -tau], [0, I]] は整数。
    for name, g in (("g1", g1), ("g2", g2)):
        Dg = np.block([[g, Z], [Z, g]])
        Binv2 = np.block([[2 * I, -t], [Z, I]])                    # 2 B^{-1}
        M2 = mm(mm(B, Dg), Binv2)
        ok = (M2 % 2 == 0).all() and (mm(mm(M2 // 2, GR), (M2 // 2).T) == GR).all()
        rep.check(f"diag({name}, {name}) preserves R and its Gram matrix", ok)
    rep.finish()


if __name__ == "__main__":
    main()
