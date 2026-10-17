"""V2. Extremal theta series, Siegel means and the derived numbers (paper §1.3, §2.3, §2.6, §4.3,
Tables 3, 5, 8, B1, B2). Exact rational arithmetic, pure Python (no numpy).
V2. 極値テータ級数、Siegel 平均と、そこから出る数（論文 §1.3, §2.3, §2.6, §4.3、表 3, 5, 8, B1, B2）。
有理数の厳密な計算。純 Python（numpy も使わない）。

Used in the proof of Theorem A: |L_6| = 15,590,400 and |L_8| = 36,957,286,800 (the numbers that
the orbit-size sums of F2 and F3 must reach). They follow from F1 because an even unimodular
lattice of dimension 56 with minimum 6 has the extremal theta series: the space of modular forms
of weight 28 has the basis E_4^7, E_4^4 Delta, E_4 Delta^2, so "1 + 0 q + 0 q^2 + ..." fixes it.
定理 A の証明に使うのは |L_6| = 15,590,400 と |L_8| = 36,957,286,800（F2・F3 の軌道の大きさの和が届くべき数）。
これは F1 から従う: 最小 6 の 56 次元偶ユニモジュラー格子のテータ級数は極値テータ級数。重さ 28 のモジュラー形式の空間の基底は
E_4^7, E_4^4 Δ, E_4 Δ^2 なので、「1 + 0q + 0q^2 + …」で 1 つに決まる。

Everything else (Siegel means, the orbit excess lambda, expected numbers of collisions) is
heuristic background and is not used in the proof.
ほか（Siegel 平均、軌道余剰 λ、衝突の期待値）は発見的な背景で、証明には使わない。

Run time: a few seconds. / 実行時間: 数秒。
Run / 実行:  python3 code/verify/v2_theta.py [--out DIR]
"""
import argparse
import os
import sys
from fractions import Fraction
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112.report import Report  # noqa: E402


# --- power series with integer coefficients, truncated below q^prec ---
# --- 整数係数の冪級数（q^prec 未満で打ち切る）---

def sigma(k, n):
    """Sum of the k-th powers of the divisors of n. / n の約数の k 乗和。"""
    return sum(d ** k for d in range(1, n + 1) if n % d == 0)


def mul(a, b, prec):
    c = [0] * prec
    for i, x in enumerate(a[:prec]):
        if x:
            for j, y in enumerate(b[:prec - i]):
                c[i + j] += x * y
    return c


def power(a, e, prec):
    r = [1] + [0] * (prec - 1)
    for _ in range(e):
        r = mul(r, a, prec)
    return r


def e4(prec):
    """E_4 = theta of E_8 = 1 + 240 sum sigma_3(m) q^m. / E_4 = E_8 のテータ級数。"""
    return [1] + [240 * sigma(3, m) for m in range(1, prec)]


def delta(prec):
    """Delta = q prod (1 - q^m)^24. / Δ = q Π(1 - q^m)^24。"""
    p = [1] + [0] * (prec - 1)
    for m in range(1, prec):
        for _ in range(24):
            for i in range(prec - 1, m - 1, -1):
                p[i] -= p[i - m]
    return [0] + p[:prec - 1]


def extremal_theta(n, prec):
    """Coefficients a_0, a_1, ... (a_m = number of vectors of norm 2m) of the extremal theta series
    of dimension n (8 | n): the unique modular form of weight n/2 equal to 1 + O(q^(floor(n/24)+1)).
    次元 n（8 | n）の極値テータ級数の係数 a_0, a_1, …（a_m = ノルム 2m のベクトルの本数）。
    重さ n/2 のモジュラー形式で 1 + O(q^(⌊n/24⌋+1)) となる唯一のもの。"""
    k, mu = n // 2, n // 24
    E, D = e4(prec), delta(prec)
    # Basis E_4^((k - 12j)/4) Delta^j = q^j + ..., triangular, so eliminate from the front.
    # 基底 E_4^((k-12j)/4) Δ^j = q^j + … は三角なので、前から消去できる。
    basis = [mul(power(E, (k - 12 * j) // 4, prec), power(D, j, prec), prec) for j in range(mu + 1)]
    f = [Fraction(x) for x in basis[0]]
    for j in range(1, mu + 1):
        c = f[j]
        f = [x - c * y for x, y in zip(f, basis[j])]
    assert all(x.denominator == 1 for x in f)
    return [int(x) for x in f]


def bernoulli(n):
    """Bernoulli number B_n (B_1 = -1/2). / ベルヌーイ数 B_n。"""
    B = [Fraction(1)]
    for m in range(1, n + 1):
        B.append(-sum(comb(m + 1, j) * B[j] for j in range(m)) / (m + 1))
    return B[n]


def siegel_mean(n, m):
    """Average number of vectors of norm 2m over the genus of even unimodular lattices of
    dimension n = 2k, weighted by 1/|Aut|: the coefficient -(2k/B_k) sigma_{k-1}(m) of E_k (§2.6).
    次元 n = 2k の偶ユニモジュラー格子の種にわたる、1/|Aut| で重みづけたノルム 2m のベクトルの本数の平均（§2.6）。"""
    k = n // 2
    return -Fraction(2 * k) / bernoulli(k) * sigma(k - 1, m)


def siegel_inner(n):
    """S(n): Siegel mean of the number of inner-shell vectors (norm < 2 floor(n/24) + 2).
    S(n): 内殻ベクトルの本数の Siegel 平均。"""
    return sum(siegel_mean(n, m) for m in range(1, n // 24 + 1))


def main():
    ap = argparse.ArgumentParser(description="V2: theta series and Siegel means")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v2_theta", a.out)

    # --- used in the proof / 証明に使うもの ---
    t56 = extremal_theta(56, 6)
    rep.value("extremal theta series, n = 56, a_0..a_5", t56)
    rep.check("n = 56: |L_6| = 15590400, |L_8| = 36957286800, |L_10| = 15284192071680",
              t56[:6] == [1, 0, 0, 15590400, 36957286800, 15284192071680])

    # --- Table B1: the same series as polynomials in A = E_4 and B = theta(Leech) = E_4^3 - 720 Delta ---
    # --- 表 B1: 同じ級数を A = E_4、B = Leech のテータ級数 = E_4^3 - 720Δ の多項式として ---
    prec = 8
    A = e4(prec)
    Bl = [x - 720 * y for x, y in zip(power(A, 3, prec), delta(prec))]
    def poly(coeffs, denom):                                                    # noqa: E306
        s = [Fraction(0)] * prec
        for (i, j), c in coeffs.items():
            term = mul(power(A, i, prec), power(Bl, j, prec), prec)
            s = [x + Fraction(c, denom) * y for x, y in zip(s, term)]
        return s
    p56 = poly({(7, 0): -159, (4, 1): 238, (1, 2): 161}, 240)
    rep.check("Table B1, n = 56: (1/240)(-159 A^7 + 238 A^4 B + 161 A B^2) is the extremal series",
              p56 == [Fraction(x) for x in extremal_theta(56, prec)])
    p112 = poly({(14, 0): 7883087, (11, 1): -18165308, (8, 2): -38110758, (5, 3): 74301892, (2, 4): 11415887}, 37324800)
    t112 = extremal_theta(112, prec)
    rep.check("Table B1, n = 112: the polynomial in A, B is the extremal series", p112 == [Fraction(x) for x in t112])
    rep.value("extremal theta series, n = 112, a_0..a_6", t112[:7])
    rep.check("n = 112: 10888335360 vectors of norm 10, 247006775232000 of norm 12",
              t112[:7] == [1, 0, 0, 0, 0, 10888335360, 247006775232000])

    # --- Table 5: Siegel means versus extremal coefficients (heuristic background) ---
    # --- 表 5: Siegel 平均と極値の係数（発見的な背景）---
    for n, ms in ((56, range(1, 6)), (112, range(1, 7))):
        rep.value(f"Table 5, n = {n}: Siegel mean of norm 2m for m = {list(ms)}",
                  [f"{float(siegel_mean(n, m)):.6g}" for m in ms])

    # --- Table B2: S(n) / 表 B2: S(n) ---
    S = {n: siegel_inner(n) for n in range(24, 137, 8)}
    rep.value("Table B2: S(n)", {n: f"{float(v):.4g}" for n, v in S.items()})
    rep.check("S(112) = 51014.2 (rounded)", round(float(S[112]), 1) == 51014.2)

    # --- Table 3: lambda = S(n) / |G| (heuristic) / 表 3: λ = S(n)/|G|（発見的）---
    rows = [("P48n", 48, 524160), ("P48p", 48, 145728), ("P48q", 48, 103776), ("P48m", 48, 1200),
            ("Gamma72", 72, 5241600), ("L80", 80, 4.5e9), ("gluings A, B", 112, 292320),
            ("R, full Aut", 112, 1169280), ("R, SL2(29) only", 112, 24360)]
    rep.value("Table 3: lambda", {name: f"{float(S[n]) / g:.2g}" for name, n, g in rows})

    # --- Table 8 / Appendix C.6: expected numbers of collisions for a random gluing (heuristic) ---
    # --- 表 8 / 付録 C.6: でたらめな貼り方での衝突の期待値（発見的）---
    odd, even = 2 ** 55 - 2 ** 27, 2 ** 55 + 2 ** 27
    L6, L8, L10 = t56[3], t56[4], t56[5]
    e66 = Fraction(L6 * L6, odd)
    e610 = 2 * Fraction(L6 * L10, odd)
    e88 = Fraction(L8 * L8, even)
    rep.value("Table 8: expected (6,6), (6,10)+(10,6), (8,8), total",
              [f"{float(e66):.3g}", f"{float(e610):.0f}", f"{float(e88):.0f}", f"{float(e66 + e610 + e88):.0f}"])
    rep.finish()


if __name__ == "__main__":
    main()
