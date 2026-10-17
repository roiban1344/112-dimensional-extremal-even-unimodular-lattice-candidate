"""S3. Estimated number of enumeration nodes for listing all vectors of norm <= 8 of R itself
(paper §6.9, "Baseline"). An estimate, not a proof; it explains why the direct check is impossible.
S3. R そのもののノルム 8 以下の全ベクトルの列挙に要る節点数の見積もり（論文 §6.9 の「基準」）。見積もりで証明ではない。
直接の検算が不可能な理由を示す。

Gaussian heuristic for Fincke-Pohst enumeration: the number of nodes at depth k is about the
volume of the k-dimensional ball of radius r divided by the volume of the projected lattice,
    N_k ~ V_k(r) / prod_{i = n-k+1}^{n} ||b_i*||,   N = sum_k N_k / 2  (one of each +-pair),
with r^2 = 8. Bases: LLL, BKZ-20 and BKZ-30 (fpylll), applied to an integer basis obtained by
scaling a 200-bit Cholesky factor of (3.1) by 2^40 (accurate enough for an estimate).
Fincke-Pohst 列挙のガウスの見積もり: 深さ k の節点数は、半径 r の k 次元球の体積を射影格子の体積で割ったもの（上の式、r^2 = 8）。
基底は LLL、BKZ-20、BKZ-30（fpylll）。(3.1) の 200 ビットの Cholesky 分解を 2^40 倍して丸めた整数基底に掛ける（見積もりには十分な精度）。

Requires SageMath (fpylll, real arithmetic). Run time: about 15 seconds.
SageMath（fpylll、実数演算）が要る。実行時間: 約 15 秒。
Run / 実行:  sage -python code/supplementary/node_estimate.py
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import data  # noqa: E402
from r112.report import Report  # noqa: E402


def estimate(A, r2):
    """Estimated nodes for the basis A (fpylll IntegerMatrix, rows) and squared radius r2.
    基底 A（fpylll の IntegerMatrix、行）と半径の 2 乗 r2 についての節点数の見積もり。"""
    from fpylll import GSO
    M = GSO.Mat(A)
    M.update_gso()
    n = A.nrows
    bstar = [math.sqrt(M.get_r(i, i)) for i in range(n)]
    total = 0.0
    for k in range(1, n + 1):
        log_ball = (k / 2) * math.log(math.pi) - math.lgamma(k / 2 + 1) + (k / 2) * math.log(r2)
        log_proj = sum(math.log(b) for b in bstar[n - k:])
        total += math.exp(log_ball - log_proj)
    return total / 2


def main():
    from fpylll import BKZ, LLL, IntegerMatrix
    from fpylll.algorithms.bkz2 import BKZReduction
    from sage.all import ZZ, RealField, matrix
    ap = argparse.ArgumentParser(description="S3: node estimate for R")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s3_node_estimate", a.out)
    GR = matrix(ZZ, data.gram_R(data.gram(a.data), data.tau(a.data)).tolist())
    rep.check("det (3.1) = 1", GR.det() == 1)
    scale = 2 ** 40
    C = GR.change_ring(RealField(200)).cholesky()             # GR = C C^T, C lower triangular
    rows = [[int((x * scale).round()) for x in row] for row in C.rows()]
    for name, block in (("LLL", None), ("BKZ-20", 20), ("BKZ-30", 30)):
        A = IntegerMatrix.from_matrix(rows)
        LLL.reduction(A)
        if block:
            BKZReduction(A)(BKZ.EasyParam(block, max_loops=4))
        rep.value(f"{name}: estimated nodes for norm <= 8 (one of each +-pair)", f"{estimate(A, 8 * scale ** 2):.2e}")
    rep.finish()


if __name__ == "__main__":
    main()
