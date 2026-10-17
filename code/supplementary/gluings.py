"""S1. The SL_2(29)-equivariant gluings of two copies of L (paper §3.3, Proposition 3.3, Table 6
rows for L; §3.3 last paragraph). Not used in the proof of Theorem A.
S1. L の 2 枚の SL_2(29)-同変な貼り方（論文 §3.3、命題 3.3、表 6 の L の行、§3.3 の最後の段落）。定理 A の証明には使わない。

Checks / 確かめること:
  (a) The F_2-linear maps of L/2L commuting with g1, g2 form an 8-dimensional space, equal to the
      image of M'/2M' (M' = certificates/commutant.npz, checked to commute with g1, g2).
      g1, g2 と可換な L/2L の F_2-線形写像は 8 次元で、M'/2M' の像に等しい。
  (b) Exactly 60 of its 2^8 elements are invertible and parity-preserving (preserve the quadratic
      form q(x + 2L) = <x, x>/2 mod 2).
      その 2^8 個の元のうち、可逆で偶奇を保つ（二次形式 q を保つ）ものはちょうど 60 個。
  (c) Under f -> u f u' (u, u' in C) and f -> f^{-1}, the 60 maps fall into 3 classes, of sizes
      6 (containing the reductions of C: forced collisions), 36 (gluing B) and 18 (gluing C,
      containing tau mod 2).
      f -> u f u'（u, u' ∈ C）と f -> f^{-1} で、60 個は 3 類に分かれる。大きさは 6（C の還元を含む: 強制された衝突）、
      36（貼り方 B）、18（貼り方 C。tau mod 2 を含む）。
  (d) M' contains exactly 24 elements t with t^2 = -5 (found among the finitely many t with
      q(t) = tr(t t*)/56 = 5); they reduce into the classes B and C, 12 each, and the given tau is
      one of them.
      M' には t^2 = -5 となる元がちょうど 24 個（q(t) = tr(t t*)/56 = 5 となる有限個の中から見つかる）。12 個ずつ類 B と類 C に落ち、
      与えた tau はその 1 つ。
The rows of Table 6 for Nebe's lattice L_{56,2}(M) are computed in supplementary/nebe_gluings.py.
表 6 の Nebe の格子 L_{56,2}(M) の行は supplementary/nebe_gluings.py で計算する。

Requires SageMath (linear algebra over F_2, PARI qfminim). Run time: about 1 minute.
SageMath（F_2 上の線形代数、PARI の qfminim）が要る。実行時間: 約 1 分。

Run / 実行:  sage -python code/supplementary/gluings.py
"""
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "produce"))
from p1_centralizer import elements_of_norm  # noqa: E402
from r112 import certs, data  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import GF, ZZ, identity_matrix, matrix
    ap = argparse.ArgumentParser(description="S1: equivariant gluings of two copies of L")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s1_gluings", a.out)
    n = 56
    F2 = GF(2)
    G = matrix(ZZ, data.gram(a.data).tolist())
    gens = [matrix(ZZ, g.tolist()) for g in data.generators(a.data)]
    tau = matrix(ZZ, data.tau(a.data).tolist())
    basis = [matrix(ZZ, X.tolist()) for X in certs.load("commutant.npz", "basis", a.certificates)]
    rep.check("the 8 basis elements of M' commute with g1, g2", len(basis) == 8 and all(X * g == g * X for X in basis for g in gens))

    # (a) commutant of g1, g2 on L/2L. / L/2L 上の g1, g2 の可換子。
    I2 = identity_matrix(F2, n)
    rows = None
    for g in gens:
        g2 = g.change_ring(F2)
        K = I2.tensor_product(g2.transpose()) - g2.tensor_product(I2)       # vec(X g - g X), row-major
        rows = K if rows is None else rows.stack(K)
    E = rows.right_kernel()
    img = matrix(F2, [X.change_ring(F2).list() for X in basis]).row_space()
    rep.value("dim of the F_2-commutant of SL_2(29) on L/2L", E.dimension())
    rep.check("(a) it is 8-dimensional and equals the image of M'/2M'", E.dimension() == 8 and E == img)

    # (b) invertible parity-preserving elements. / 可逆で偶奇を保つ元。
    Gb = G.change_ring(F2)
    q0 = [int(G[i, i]) // 2 % 2 for i in range(n)]

    def parity_preserving(f):
        # f preserves the bilinear form mod 2 and q on a basis; then it preserves q everywhere
        # (q(x + y) = q(x) + q(y) + <x, y> mod 2).
        # f が 2 を法とした双線形形式と基底上の q を保てば、q 全体を保つ（q(x+y) = q(x) + q(y) + <x,y>）。
        if f * Gb * f.transpose() != Gb:
            return False
        for i in range(n):
            v = f.row(i).change_ring(ZZ)
            if (v * G * v) // 2 % 2 != q0[i]:
                return False
        return True
    Eb = [matrix(F2, n, n, list(v)) for v in E.basis()]
    maps = []
    for coeffs in itertools.product([0, 1], repeat=8):
        f = sum((c * b for c, b in zip(coeffs, Eb)), matrix(F2, n, n))
        if f.is_invertible() and parity_preserving(f):
            f.set_immutable()
            maps.append(f)
    rep.value("invertible parity-preserving elements", len(maps))
    rep.check("(b) exactly 60", len(maps) == 60)

    # (c) classes under u f u' and inversion. / u f u' と逆元による類。
    half = elements_of_norm(basis, G, 1)
    C2 = []
    for X in half:
        Xb = X.change_ring(F2)
        Xb.set_immutable()
        if Xb not in C2:
            C2.append(Xb)                                    # -1 = 1 mod 2 / 2 を法として -1 = 1
    rep.value("|C mod 2|", len(C2))
    todo = set(maps)
    classes = []
    while todo:
        f0 = min(todo, key=lambda f: tuple(f.list()))
        orb = set()
        for u in C2:
            for v in C2:
                for f in (f0, f0.inverse()):
                    h = u * f * v
                    h.set_immutable()
                    orb.add(h)
        classes.append(orb)
        todo -= orb
    tau2 = tau.change_ring(F2)
    tau2.set_immutable()
    info = []
    for orb in classes:
        info.append({"size": len(orb), "contains C": any(u in orb for u in C2), "contains tau": tau2 in orb})
    info.sort(key=lambda d: d["size"])
    rep.value("(c) classes", info)
    rep.check("(c) three classes: 6 (containing C), 36, 18 (containing tau)",
              [(d["size"], d["contains C"], d["contains tau"]) for d in info] == [(6, True, False), (18, False, True), (36, False, False)])

    # (d) elements with t^2 = -5. / t^2 = -5 となる元。
    I = identity_matrix(ZZ, n)
    cand = elements_of_norm(basis, G, 5)
    taus = [s * t for t in cand for s in (1, -1) if t * t == -5 * I]
    rep.value("elements of M' with q = 5 (up to sign)", len(cand))
    rep.check("(d) exactly 24 elements t of M' with t^2 = -5, and tau is one of them", len(taus) == 24 and tau in taus)
    where = {}
    for t in taus:
        t2 = t.change_ring(F2)
        t2.set_immutable()
        k = next(len(orb) for orb in classes if t2 in orb)
        where[k] = where.get(k, 0) + 1
    rep.value("(d) number of these t in the class of size 36 (B) and 18 (C)", where)
    rep.check("(d) 12 in class B and 12 in class C", where == {36: 12, 18: 12})
    rep.finish()


if __name__ == "__main__":
    main()
