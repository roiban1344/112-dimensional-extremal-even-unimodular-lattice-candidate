"""S15. Floating-point enumerations (paper §6.1 (c), §6.6, §6.7 "CVP"; Table D2 [C7], [C12]). These are
supplementary: fplll and PARI use floating-point Gram-Schmidt data and do not guarantee completeness, so
the results are not part of the proof (which uses verify/v1, v5, v6, v7).
S15. 浮動小数点の列挙（論文 §6.1 (c)、§6.6、§6.7 の CVP。表 D2 の [C7], [C12]）。補助: fplll と PARI は浮動小数点の Gram-Schmidt を使い完全性を
保証しないので、結果は証明の一部ではない（証明は verify/v1, v5, v6, v7）。

Modes / モード:
  svp      all vectors of L of norm <= 4.5 by fplll's enumeration, in two bases (BKZ-40 of the given basis,
           and of a random unimodular transform of it). Expected: none.
           fplll の列挙で L のノルム 4.5 以下の全ベクトルを、2 つの基底で（与えた基底の BKZ-40 と、乱数のユニモジュラー変換の BKZ-40）。期待: 無し。
  qfminim  the same with PARI's qfminim (norm <= 4) on a BKZ-50 basis. Expected: none.
           同じことを BKZ-50 基底で PARI の qfminim（ノルム 4 以下）。期待: 無し。
  cvp610   for the 656 classes r_j tau + 2L: the shortest element of norm <= 10, by closest-vector
           enumeration (squared radius 2.8 in a BKZ-45 basis of a random transform). Expected: none.
           656 個の類 r_j tau + 2L について、ノルム 10 以下の最短の元を最近ベクトル列挙で（乱数変換の BKZ-45 基底、半径² 2.8）。期待: 無し。
  cvp88    for the pairs (x_i, u_k): does x_i u_k tau + 2L contain a vector of norm 8? Closest-vector
           enumeration at squared radius 2.05 (a norm-8 element is at squared distance exactly 2, the next
           possible norm 12 at 3). Split into parts with --part/--nparts, resumable. Expected: none. With
           --tau B the same for gluing B (expected: the 10 colliding pairs of supplementary/gluing_b.py).
           組 (x_i, u_k) について、x_i u_k tau + 2L はノルム 8 のベクトルを含むか。半径² 2.05 の最近ベクトル列挙（ノルム 8 の元は距離² ちょうど 2、
           次にありうるノルム 12 は 3）。--part/--nparts で分割、再開可能。期待: 無し。--tau B で貼り方 B（期待: supplementary/gluing_b.py の衝突 10 組）。

The class t + 2L contains a vector of norm <= m iff some y in L has <y + t/2, y + t/2> <= m/4: a
closest-vector problem with target -t/2. The target modulo L has coordinates 0 or 1/2 in the reduced basis.
類 t + 2L がノルム m 以下の元を含む ⟺ <y + t/2, y + t/2> ≤ m/4 となる y ∈ L がある。目標 -t/2 の最近ベクトル問題。簡約基底では目標の座標は 0 か 1/2。

Requires SageMath (fpylll, PARI). Run times: svp 10-30 min per basis, qfminim about 30 min, cvp610
about 2.5 hours (13 s per class), cvp88 about 4 x 9 hours for all pairs (4 parts).
SageMath（fpylll, PARI）が要る。実行時間: svp 1 基底 10〜30 分、qfminim 約 30 分、cvp610 約 2.5 時間（1 類 13 秒）、cvp88 は全組で約 4 × 9 時間（4 分割）。
Run / 実行:  sage -python code/supplementary/float_enumerations.py <svp|qfminim|cvp610|cvp88> [--part P --nparts Q] [--tau B]
"""
import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from r112 import certs, data, exact  # noqa: E402
from r112.report import Report  # noqa: E402


def random_unimodular(n, rng, steps=4000):
    """A product of random elementary row operations (a change of basis). / 乱数の基本変形の積（基底の取り替え）。"""
    from sage.all import ZZ, identity_matrix
    T = identity_matrix(ZZ, n)
    for _ in range(steps):
        i, j = rng.randrange(n), rng.randrange(n)
        if i != j:
            T.add_multiple_of_row(i, j, rng.choice((-1, 1)))
    return T


def bkz_basis(G, block, loops=8):
    """U with U G U^T BKZ-reduced (fpylll on the Gram matrix). / U G U^T が BKZ 簡約になる U。"""
    from fpylll import BKZ, GSO, LLL, IntegerMatrix
    from fpylll.algorithms.bkz2 import BKZReduction
    from sage.all import ZZ, matrix
    n = G.nrows()
    A = IntegerMatrix.from_matrix([[int(v) for v in r] for r in G.rows()])
    U = IntegerMatrix.identity(n)
    M = GSO.Mat(A, U=U, gram=True, float_type="mpfr", flags=GSO.INT_GRAM)
    M.update_gso()
    LLL.Reduction(M)()
    for bs in range(20, block + 1, 10):
        BKZReduction(M)(BKZ.Param(block_size=bs, max_loops=loops, strategies=BKZ.DEFAULT_STRATEGY))
    return matrix(ZZ, n, n, lambda i, j: U[i, j])


class CosetMinimum:
    """Shortest element of a class w + 2L by closest-vector enumeration (fplll).
    類 w + 2L の最短の元を最近ベクトル列挙（fplll）で求める。"""

    def __init__(self, G, T):
        from fpylll import GSO, IntegerMatrix
        from sage.all import ZZ, matrix
        self.n = G.nrows()
        self.G = np.array(G.rows(), dtype=np.int64)
        self.T = np.array(T.rows(), dtype=np.int64)                     # reduced basis = T * original
        self.Tinv = np.array(T.inverse().change_ring(ZZ).rows(), dtype=np.int64)
        Gr = T * G * T.transpose()
        self.Gr_diag = sorted(set(int(v) for v in Gr.diagonal()))
        Ar = IntegerMatrix.from_matrix([[int(v) for v in r] for r in Gr.rows()])
        self.M = GSO.Mat(Ar, gram=True, float_type="d", flags=GSO.INT_GRAM)
        self.M.update_gso()
        self.mu = np.array([[self.M.get_mu(i, j) if j < i else 0.0 for j in range(self.n)] for i in range(self.n)])

    def shortest_in_class(self, w, radius2):
        """An element of w + 2L of norm <= 4 radius2 (the shortest found), or None if the enumeration finds none.
        w + 2L のノルム 4 radius2 以下の元（見つかった最短）。列挙が見つけなければ None。"""
        from fpylll import Enumeration, EnumerationError
        tr = (w @ self.Tinv) % 2
        tau = tr / 2.0
        try:
            sol = Enumeration(self.M).enumerate(0, self.n, radius2, 0, target=list(tau + tau @ self.mu))
        except EnumerationError:
            return None
        y = np.array([int(round(c)) for c in sol[0][1]], dtype=np.int64)
        v = (tr - 2 * y) @ self.T
        assert ((v - w) % 2 == 0).all()
        return v


def main():
    from fpylll import Enumeration, EnumerationError, GSO, IntegerMatrix
    from sage.all import ZZ, identity_matrix, matrix, pari
    ap = argparse.ArgumentParser(description="S15: floating-point enumerations")
    ap.add_argument("mode", choices=("svp", "qfminim", "cvp610", "cvp88"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--tau", choices=("C", "B"), default="C")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = f"s15_{a.mode}" + (f"_part{a.part}of{a.nparts}" if a.mode == "cvp88" else "") + ("_B" if a.tau == "B" else "")
    rep = Report(tag, a.out)
    n = 56
    G = matrix(ZZ, data.gram(a.data).tolist())
    tn = data.tau(a.data) if a.tau == "C" else certs.load("negative_control_B.npz", "tau_class_B", a.certificates).astype(np.int64)

    if a.mode == "svp":
        for trial in range(2):
            T0 = random_unimodular(n, random.Random(100 + trial)) if trial else identity_matrix(ZZ, n)
            T = bkz_basis(T0 * G * T0.transpose(), 40)
            Gr = T * T0 * G * (T * T0).transpose()
            A = IntegerMatrix.from_matrix([[int(v) for v in r] for r in Gr.rows()])
            M = GSO.Mat(A, gram=True, float_type="ld", flags=GSO.INT_GRAM)
            M.update_gso()
            try:
                sol = Enumeration(M).enumerate(0, n, 4.5, 0)
                rep.check(f"basis {trial}: no vector of norm <= 4.5 (found one of norm {sol[0][0]})", False)
            except EnumerationError:
                rep.check(f"basis {trial} (reduced diagonal {sorted(set(int(v) for v in Gr.diagonal()))}): fplll finds no nonzero vector of norm <= 4.5", True)
    elif a.mode == "qfminim":
        T = bkz_basis(G, 50)
        Gr = T * G * T.transpose()
        assert abs(T.det()) == 1
        r = pari(Gr).qfminim(4, 1, 0)
        rep.check("PARI qfminim on a BKZ-50 basis: no nonzero vector of norm <= 4", int(r[0]) == 0)
    elif a.mode == "cvp610":
        R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
        T0 = random_unimodular(n, random.Random(7))
        cm = CosetMinimum(G, bkz_basis(T0 * G * T0.transpose(), 45) * T0)
        rep.value("reduced diagonal of the basis", cm.Gr_diag)
        found = 0
        for j, r in enumerate(R6):
            v = cm.shortest_in_class(exact.i64_matmul(r, tn), 2.8)          # squared distances are multiples of 1/2
            if v is not None and int(v @ cm.G @ v) <= 10:
                found += 1
            if (j + 1) % 50 == 0:
                rep.log(f"{j + 1} / {len(R6)} classes, found {found}")
        rep.value("classes r_j tau + 2L with an element of norm <= 10 found by CVP", found)
        rep.check("none of the 656" if a.tau == "C" else "(gluing B) count recorded", found == 0 or a.tau == "B")
    else:
        X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
        U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
        cm = CosetMinimum(G, bkz_basis(G, 40))
        total = 6 * len(X)
        idxs = range(a.part, total, a.nparts)
        ck = os.path.join(rep.out_dir, tag + ".state.json")
        st = json.load(open(ck)) if os.path.exists(ck) else {"done": 0, "found": []}
        rep.log(f"part {a.part}/{a.nparts}: {len(idxs)} of {total} pairs; resuming at {st['done']}")
        for pos in range(st["done"], len(idxs)):
            p = idxs[pos]
            i, k = divmod(p, 6)
            v = cm.shortest_in_class(exact.i64_matmul(exact.i64_matmul(X[i], U[k]), tn), 2.05)
            if v is not None:
                assert int(v @ cm.G @ v) == 8
                st["found"].append(int(p))
            if (pos + 1) % 5000 == 0 or pos + 1 == len(idxs):
                st["done"] = pos + 1
                tmp = ck + ".tmp"
                json.dump(st, open(tmp, "w"))
                os.replace(tmp, ck)
                rep.log(f"{pos + 1} / {len(idxs)} pairs, found {len(st['found'])}")
        rep.value("pairs whose class contains a norm-8 vector (found by CVP)", st["found"])
        rep.check("none" if a.tau == "C" else "(gluing B) pairs recorded", not st["found"] or a.tau == "B")
    rep.finish()


if __name__ == "__main__":
    main()
