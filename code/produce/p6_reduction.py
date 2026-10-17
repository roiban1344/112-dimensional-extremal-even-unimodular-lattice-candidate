"""Produce the two basis-change certificates (paper §6.1 step 1, Appendix D).
検算に使う 2 つの基底変換を作る（論文 §6.1 の手順 1、付録 D）。

  certificates/reduction_L.txt   U (56 x 56) with U G U^T BKZ-40 reduced; used by verify/v1_min_L.py
                                 U G U^T が BKZ-40 簡約になる U。verify/v1_min_L.py が使う
  certificates/reduction_R.txt   U (112 x 112) with U (3.1) U^T = data/R112_gram_LLL.txt
                                 U (3.1) U^T = data/R112_gram_LLL.txt となる U

How they are found does not matter for the proof: the verification only checks that U is
an integer matrix of determinant +-1 (and, for R, the matrix identity).
作り方は証明に関係しない。検算は U が行列式 ±1 の整数行列であること（R については行列の等式も）だけを確かめる。

Method / 方法:
  L: the Gram matrix is turned into an integer basis by scaling a floating-point Cholesky
     factor by 2^40 and rounding; fpylll's LLL and BKZ (block size 40, at most 4 tours)
     are applied to it, and only the accumulated transformation is kept.
     Gram 行列の浮動小数点の Cholesky 分解を 2^40 倍して丸めた整数基底に、fpylll の LLL と BKZ（ブロック 40、最大 4 周）を掛け、
     変換行列だけを使う。
  R: PARI's qflllgram on the Gram matrix (3.1) (this is how data/R112_gram_LLL.txt was made).
     (3.1) の Gram 行列に PARI の qflllgram（data/R112_gram_LLL.txt はこれで作った）。

Requires SageMath (PARI, fpylll). Run time about 10 minutes (BKZ-40).
SageMath（PARI, fpylll）が要る。実行時間は約 10 分（BKZ-40）。

Run / 実行:  sage -python code/produce/p6_reduction.py [--data DIR] [--out DIR]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact  # noqa: E402


def bkz_transform(G, block, loops=4):
    """U with U G U^T BKZ-reduced (row convention). / U G U^T が BKZ 簡約になる U（行の約束）。"""
    from fpylll import BKZ, GSO, LLL, IntegerMatrix
    from fpylll.algorithms.bkz2 import BKZReduction
    n = len(G)
    C = np.linalg.cholesky(np.asarray(G, dtype=float))          # G = C C^T; rows of C are basis vectors / C の行が基底
    A = IntegerMatrix.from_matrix([[int(round(v * 2 ** 40)) for v in row] for row in C])
    U = IntegerMatrix.identity(n)
    M = GSO.Mat(A, U=U, float_type="mpfr")
    LLL.Reduction(M)()
    BKZReduction(M)(BKZ.EasyParam(block, max_loops=loops))
    return np.array([[U[i, j] for j in range(n)] for i in range(n)], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="directory of the four input files / 入力ファイルの置き場所")
    ap.add_argument("--out", default=certs.DEFAULT_DIR, help="output directory / 出力先")
    ap.add_argument("--block", type=int, default=40)
    a = ap.parse_args()
    t0 = time.time()
    G = data.gram(a.data)
    t = data.tau(a.data)

    # L: BKZ-40 transformation. / L: BKZ-40 の変換。
    U = bkz_transform(G, a.block)
    assert abs(exact.det_bareiss(U)) == 1
    Gr = exact.i64_matmul(exact.i64_matmul(U, G), U.T)
    certs.save_matrix(os.path.join(a.out, "reduction_L.txt"), U,
                      f"U (56 x 56, rows = new basis in terms of the old): U G U^T is BKZ-{a.block} reduced.\n"
                      f"U（56 × 56、行 = 新しい基底を古い基底で表したもの）: U G U^T は BKZ-{a.block} 簡約。\n"
                      f"diagonal of U G U^T / U G U^T の対角: {sorted(set(int(v) for v in np.diag(Gr)))}")
    print(f"reduction_L.txt: BKZ-{a.block}, diagonal norms {sorted(set(int(v) for v in np.diag(Gr)))}  [{time.time() - t0:.0f} s]", flush=True)

    # R: qflllgram, compared with the distributed reduced Gram matrix.
    # R: qflllgram。配布した簡約 Gram 行列と比べる。
    from sage.all import ZZ, matrix, pari
    B = data.gram_R(G, t)
    V = pari(matrix(ZZ, B.tolist())).qflllgram().sage()        # columns = new basis (PARI convention) / 列が新しい基底
    UR = np.array(V.transpose().rows(), dtype=np.int64)        # rows = new basis / 行が新しい基底
    assert (exact.i64_matmul(exact.i64_matmul(UR, B), UR.T) == data.gram_R_reduced(a.data)).all()
    certs.save_matrix(os.path.join(a.out, "reduction_R.txt"), UR,
                      "U (112 x 112): U [[3G, tau G], [(tau G)^T, 2G]] U^T = data/R112_gram_LLL.txt (PARI qflllgram).\n"
                      "U（112 × 112）: U [[3G, tau G], [(tau G)^T, 2G]] U^T = data/R112_gram_LLL.txt（PARI の qflllgram）。")
    print(f"reduction_R.txt: matches data/R112_gram_LLL.txt  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
