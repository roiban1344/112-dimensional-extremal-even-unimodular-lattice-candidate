"""S2. Automorphisms of R preserving the standard frame, and the halving of the (8,8) pairs in
route B (paper §6.9, Table 13a). Not used in the proof of Theorem A.
S2. 標準の枠を保つ R の自己同型と、経路 B での (8,8) 型の組の半減（論文 §6.9、表 13a）。定理 A の証明には使わない。

For a, d in the centralizer C, the map (x, v) -> (x a, v d) preserves R = {(x, v) : v = x tau mod 2L}
iff a tau = tau d (mod 2). Then the test of the pair (x_i, u) (does x_i u tau + 2L contain a norm-8
vector?) is equivalent to that of (x_i, u a). So the number of pairs to test is divided by
|S / {+-1}|, S = { a in C : a tau = tau d (mod 2) for some d in C }.
C の元 a, d について、(x, v) -> (x a, v d) が R を保つ ⟺ a tau ≡ tau d (mod 2)。このとき組 (x_i, u) の検査は (x_i, u a) の検査と同値。
よって検査する組の数は |S/{±1}| で割れる。S = { a ∈ C : ある d ∈ C で a tau ≡ tau d (mod 2) }。

Only numpy. Run time: seconds. / numpy のみ。実行時間: 数秒。
Run / 実行:  python3 code/supplementary/frame_stabilizer.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="S2: frame stabilizer")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s2_frame_stabilizer", a.out)
    mm = exact.i64_matmul
    G = data.gram(a.data)
    t = data.tau(a.data)
    half = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    C = list(half) + [-u for u in half]
    rep.check("C: 12 isometries of L", len(C) == 12 and all((mm(mm(u, G), u.T) == G).all() for u in C))
    comm = sum((mm(u, t) == mm(t, u)).all() for u in C)
    anti = sum((mm(u, t) == -mm(t, u)).all() for u in C)
    rep.value("elements of C commuting with tau exactly", int(comm))
    rep.value("elements of C anticommuting with tau exactly", int(anti))
    S = [u for u in C if any(((mm(u, t) - mm(t, d)) % 2 == 0).all() for d in C)]
    rep.value("|S|", len(S))
    rep.value("pairs needed in route B: 1527984 / |S/{+-1}|", 1527984 // (len(S) // 2))
    rep.check("|S/{+-1}| = 2, so route B needs 763992 pairs", len(S) // 2 == 2)
    rep.finish()


if __name__ == "__main__":
    main()
