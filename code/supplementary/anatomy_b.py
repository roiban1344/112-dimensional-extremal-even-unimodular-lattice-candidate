"""S16. Support modules of the norm-8 vectors of gluing B (paper §7.4, Table 14; Table D2 [C19]).
Not used in the proof of Theorem A.
S16. 貼り方 B のノルム 8 のベクトルの台加群（論文 §7.4、表 14。表 D2 の [C19]）。定理 A の証明には使わない。

Input: the 10 colliding pairs (x, v) of gluing B from supplementary/gluing_b.py (out/gluing_b_collisions.npz).
Each gives two norm-8 vectors of R_B, z = (x, v)/sqrt2 and z' = (v, x)/sqrt2 (the swap preserves R_B because
tau_B^2 = 1 mod 2), 20 in all. Writing z = e_1 tensor x + e_2 tensor y with y = (v - x tau_B)/2, the
support module is N_z = O x + O y (O = Z[sqrt -5] acting by sqrt -5 -> tau_B).
入力: supplementary/gluing_b.py の衝突 10 組 (x, v)（out/gluing_b_collisions.npz）。各組から R_B のノルム 8 のベクトル z = (x, v)/√2 と
z' = (v, x)/√2（tau_B^2 ≡ 1 mod 2 なので入れ替えは R_B を保つ）、合わせて 20 本。z = e_1 ⊗ x + e_2 ⊗ y（y = (v - x tau_B)/2）と書くと、
台加群は N_z = O x + O y（O = Z[√-5] は √-5 -> tau_B で作用）。

For each z: the Gram matrix of (x, y) for the trace-normalised Hermitian form
    k(a, b) = (1/2)[<a, b> + <a tau, b> / sqrt(-5)]   (so <a, a> = 2 k(a, a)),
its determinant det_k, Coulangeon's bound 2 sqrt(det h_H det_k N) <= k_R(z, z) = 4, the index of
N_z in its saturation (as Z-lattices), and the minimum of N_z (exact: the vector x has norm 8, and the
enumeration that cannot miss vectors finds none of norm <= 6). Expected (Table 14): two types,
det_k = 4 (12 vectors; Gram matrix 2 H^{-1}, equality in Coulangeon's bound) and det_k = 19/5 (8 vectors),
both saturated (index 1) and with minimum 8.
各 z について: トレースで正規化したエルミート形式 k による (x, y) の Gram 行列、その行列式 det_k、Coulangeon の評価、飽和の中での N_z の指数
（Z-格子として）、N_z の最小（厳密: x はノルム 8、取りこぼしの無い列挙でノルム 6 以下は無い）。期待値（表 14）: 2 つの型、det_k = 4（12 本。
Gram 行列は 2H^{-1} で Coulangeon の評価の等号）と det_k = 19/5（8 本）。どちらも飽和（指数 1）で最小 8。

Requires SageMath (the field Q(sqrt -5)). Run time: about 1 minute.
SageMath（体 Q(√-5)）が要る。実行時間: 約 1 分。
Run / 実行:  sage -python code/supplementary/anatomy_b.py
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from r112 import data  # noqa: E402
from r112 import enumerate as exact_enum  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import QQ, ZZ, QuadraticField, matrix, sqrt, vector
    ap = argparse.ArgumentParser(description="S16: support modules of gluing B")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s16_anatomy_b", a.out)
    G = matrix(ZZ, data.gram(a.data).tolist())
    with np.load(os.path.join(rep.out_dir, "gluing_b_collisions.npz"), allow_pickle=False) as z:
        X, V, tB = z["x"], z["v"], matrix(ZZ, z["tau_B"].tolist())
    rep.value("colliding pairs from supplementary/gluing_b.py", len(X))
    K = QuadraticField(-5, "s")
    s = K.gen()

    def k(u, w):
        """k(u, w) = (1/2)[<u, w> + <u tau, w> / sqrt(-5)] in K. / K に値をとる k(u, w)。"""
        return (QQ(u * G * w) + QQ(u * tB * G * w) / s) / 2

    H = matrix(K, [[3, s], [-s, 2]])
    build = os.path.join(rep.out_dir, "build")
    table = {}
    for x0, v0 in zip(X, V):
        for xa, va in ((x0, v0), (v0, x0)):                              # z and the swapped z'
            x, v = vector(ZZ, xa.tolist()), vector(ZZ, va.tolist())
            yv = v - x * tB
            assert all(c % 2 == 0 for c in yv) and (x * G * x + v * G * v) / 2 == 8
            y = yv / 2
            Gk = matrix(K, [[k(x, x), k(x, y)], [k(y, x), k(y, y)]])
            dk = Gk.det()
            assert dk in QQ
            dk = QQ(dk)
            # Coulangeon: k_R(z, z) = 4 >= 2 sqrt(det h_H * det_k N_z), det h_H = 1.
            # Coulangeon: k_R(z, z) = 4 ≥ 2√(det h_H · det_k N_z)、det h_H = 1。
            assert 4 >= 2 * sqrt(dk) and H.det() == 1
            N0 = matrix(ZZ, [x, x * tB, y, y * tB])
            assert N0.rank() == 4
            N = N0.hermite_form(include_zero_rows=False).saturation()
            GN = N * G * N.transpose()
            # [N : N0]^2 = det(Gram of N0) / det(Gram of N). / 指数の 2 乗は Gram 行列式の比。
            index = sqrt(QQ((N0 * G * N0.transpose()).det()) / QQ(GN.det()))
            found, _ = exact_enum.short_vectors([[int(c) for c in r] for r in GN.rows()], 6, build)
            mn = 8 if not found else min(m for _, m in found)
            key = f"det_k {dk}, index {index}, minimum {mn}"
            table[key] = table.get(key, 0) + 1
    rep.value("(det_k N_z, index of N_z in its saturation, minimum of N_z) -> number of vectors", table)
    rep.check("Table 14: two types, det_k = 4 and det_k = 19/5, all saturated with minimum 8",
              set(table) == {"det_k 4, index 1, minimum 8", "det_k 19/5, index 1, minimum 8"})
    rep.value("2 H^{-1} (the Gram matrix of type I up to reduction) and its determinant",
              [str(2 * H.inverse()), str((2 * H.inverse()).det())])
    rep.finish()


if __name__ == "__main__":
    main()
