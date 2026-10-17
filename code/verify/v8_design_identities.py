"""V8. Completeness test by spherical design identities (paper §6.3). Not needed for Theorem A:
F2 and F3 are already proved by the orbit-size sums (V3, V4); this is an independent consistency test.
V8. 球面デザインの恒等式による完全性の検査（論文 §6.3）。定理 A には不要: F2・F3 は軌道の大きさの和（V3, V4）で
すでに示される。これは独立な整合性の検査。

Each nonempty shell of an extremal lattice of dimension 24k + 8 is a spherical 7-design
(Venkov), so for every vector a
24k + 8 次元の極値格子の空でない殻は球面 7-デザイン（Venkov）なので、任意のベクトル a について
    sum_{x in L_m} <x, a>^2 = |L_m| m <a, a> / 56,
    sum_{x in L_m} <x, a>^4 = |L_m| 3 m^2 <a, a>^2 / (56 * 58),
    sum_{x in L_m} <x, a>^6 = |L_m| 15 m^3 <a, a>^3 / (56 * 58 * 60).
The left sides are computed from the orbit lists: for an orbit O = x G,
左辺を軌道の一覧から計算する。軌道 O = x G について
    sum_{y in O} <y, a>^k = (|O| / |G|) sum_{g in G} <x g, a>^k = (|O| / |G|) * 2 sum_{s in G/{+-1}} <x s, a>^k
(k even). For m = 6 the orbits are those of min_reps.npz with their sizes (V3); for m = 8 one
vector per orbit is taken from out/K8_first.npy (V4), all orbits being regular.
A duplicated or missing orbit would break the identities (barring coincidence).
（k は偶数）。m = 6 では min_reps.npz の軌道とその大きさ（V3）、m = 8 では out/K8_first.npy（V4）から軌道ごとに 1 本を取る
（どの軌道も正則）。軌道の重複や欠落があれば、（偶然を除いて）等式が崩れる。

Exactness: inner products in float64 with the exactness bound asserted; powers and sums in int64
per block (bounded), accumulated in Python integers; comparison with exact fractions.
厳密さ: 内積は float64 で厳密さの上界を確かめる。冪と和はブロックごとに int64（上界を確かめる）で、Python の整数に足し込む。
比較は厳密な分数で。

Run time: about 10 minutes (after V4). Memory: about 2 GB.
実行時間: 約 10 分（V4 のあと）。メモリ: 約 2 GB。

Run / 実行:  python3 code/verify/v8_design_identities.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import sys
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, mod2  # noqa: E402
from r112.report import Report  # noqa: E402

N6, N8 = 15590400, 36957286800


def power_sums(X, weights, B):
    """sum_r weights[r] * sum_s <x_r s, a>^k for k = 2, 4, 6, where column s of B is s G a^T.
    Returns three Python integers. / k = 2, 4, 6 について Σ_r weights[r] Σ_s <x_r s, a>^k（B の第 s 列は s G a^T）。"""
    tot = [0, 0, 0]
    for i in range(0, len(X), 256):
        Xb = X[i:i + 256]
        P = exact.f64_matmul(Xb, B)                          # (block, |S|): <x s, a>
        pmax = int(np.abs(P).max())
        assert pmax ** 6 * P.shape[1] < 2 ** 62, "int64 power sum could overflow"
        P2 = P * P
        for j, Pk in enumerate((P2, P2 * P2, P2 * P2 * P2)):
            rows = Pk.sum(axis=1)                             # per representative, exact in int64
            tot[j] += sum(int(w) * int(v) for w, v in zip(weights[i:i + 256], rows))
    return tot


def main():
    ap = argparse.ArgumentParser(description="V8: spherical design identities")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v8_design_identities", a.out)

    G = data.gram(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data)).astype(np.int64)
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
    U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    first = np.load(os.path.join(rep.out_dir, "K8_first.npy"))
    rep.check("one vector for each of the 1517130 orbits of L_8 (from V4)", len(first) == 1517130)
    X8 = exact.i64_matmul(X[first // 6][:, None, :], U[first % 6])[:, 0, :]     # x_i u_k
    # Orbit sizes of L_6: 2 |orbit of the class| (as in V3). / L_6 の軌道の大きさ（V3 と同じ）。
    _, size6 = mod2.Canonical(S).canon_and_orbit_size(mod2.pack(R6))
    size6 = 2 * size6
    rep.check("orbit sizes of L_6 add up to 15590400", int(size6.sum()) == N6)

    rng = np.random.default_rng(20261013)
    ok = True
    for trial in range(3):
        av = np.eye(56, dtype=np.int64)[0] if trial == 0 else rng.integers(-1, 2, size=56)
        na = int(av @ G @ av)
        B = exact.i64_matmul(exact.i64_matmul(S, G), av).T                       # (56, |S|): column s = s G a^T
        for m, Xm, w, N in ((6, R6, size6, N6), (8, X8, np.full(len(X8), 24360), N8)):
            sums = power_sums(Xm, w, B)
            for (k, c, d), s in zip(((2, 1, 56), (4, 3, 56 * 58), (6, 15, 56 * 58 * 60)), sums):
                lhs = Fraction(2 * s, 24360)                   # (|O| / |G|) * 2 sum over G/{+-1}
                rhs = Fraction(N * c * m ** (k // 2) * na ** (k // 2), d)
                good = lhs == rhs
                ok &= good
                rep.log(f"a#{trial} (<a,a> = {na}), m = {m}, degree {k}: sum = {lhs}, design value = {rhs}, equal: {good}")
    rep.check("all 18 identities hold exactly (3 vectors a, m = 6 and 8, degrees 2, 4, 6)", ok)
    rep.finish()


if __name__ == "__main__":
    main()
