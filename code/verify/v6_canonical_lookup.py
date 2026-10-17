"""V6. F5b by canonical forms: no class x_i u_k tau + 2L contains a vector of norm 8
(paper §4.4 F5b, §6.5, Proposition 6.3, Table 10). Also the type (6,6) by the same method.
V6. 標準形による F5b: 類 x_i u_k tau + 2L はどれもノルム 8 のベクトルを含まない
（論文 §4.4 の F5b、§6.5、命題 6.3、表 10）。あわせて型 (6,6) も同じ方法で。

Proposition 6.3 (assuming F3): a class c contains a vector of norm 8 iff can(c) is in
K_8 = { can(x_i u_k + 2L) }. V4 computes K_8 and writes it to out/K8.npy; this script reads it.
For every one of the 1,527,984 pairs (i, k) we compute can(x_i u_k tau + 2L) and look it up.
命題 6.3（F3 を仮定）: 類 c がノルム 8 のベクトルを含む ⟺ can(c) ∈ K_8 = { can(x_i u_k + 2L) }。
K_8 は V4 が計算して out/K8.npy に書く。このスクリプトはそれを読む。1,527,984 組 (i, k) のすべてについて
can(x_i u_k tau + 2L) を計算して表を引く。

Similarly, with K_6 = { can(r_j + 2L) } (F2), no class r_j tau + 2L contains a vector of norm 6
(type (6,6); F5a already covers it, this is a second method).
同様に K_6 = { can(r_j + 2L) }（F2）について、類 r_j tau + 2L はノルム 6 のベクトルを含まない（型 (6,6)。F5a ですでに示されるが、別の方法）。

Positive control: for tau replaced by the identity every class r_j + 2L is in K_6 and every
x_i u_k + 2L is in K_8.
陽性対照: tau を恒等写像に置き換えると、すべての r_j + 2L が K_6 に、すべての x_i u_k + 2L が K_8 に入る。

Exactness: integer arithmetic only. / 厳密さ: 整数演算のみ。
Run time: about 10 minutes (after V4). Memory: about 1.5 GB.
実行時間: 約 10 分（V4 のあと）。メモリ: 約 1.5 GB。

Run / 実行:  python3 code/verify/v6_canonical_lookup.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, mod2  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V6: F5b by canonical forms")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v6_canonical_lookup", a.out)

    t = data.tau(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
    U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    K8 = np.load(os.path.join(rep.out_dir, "K8.npy"))
    rep.check("K_8 from V4 has 1517130 entries", len(K8) == 1517130)
    mm = exact.i64_matmul

    can = mod2.Canonical(S)
    can.self_test(S, np.random.default_rng(3))
    rep.check("tau commutes with g1, g2 modulo 2 (so can(x tau) depends only on the orbit of x)",
              all(((mm(g, t) - mm(t, g)) % 2 == 0).all() for g in data.generators(a.data)))

    # Type (6,6) and the positive control on L_6. / 型 (6,6) と L_6 での陽性対照。
    K6 = np.unique(can.canon(mod2.pack(R6)))
    c66 = can.canon(mod2.pack(mm(R6, t)))
    rep.check("positive control: every r_j + 2L is in K_6", np.isin(can.canon(mod2.pack(R6)), K6).all())
    rep.value("(6,6): classes r_j tau + 2L containing a norm-6 vector", int(np.isin(c66, K6).sum()))
    rep.check("(6,6): none of the 656", not np.isin(c66, K6).any())
    rep.check("consistency: no r_j tau + 2L is in K_8 (it is an odd class)", not np.isin(c66, K8).any())

    # Type (8,8). / 型 (8,8)。
    hits, ctrl, total = 0, 0, 0
    step = 1 << 14
    for k, u in enumerate(U):
        XU = mm(X, u)
        for i in range(0, len(XU), step):
            B = XU[i:i + step]
            f = can.canon(mod2.pack(mm(B, t)))
            hits += int(np.isin(f, K8).sum())
            if i == 0:                                       # positive control on one block per k
                ctrl += int(np.isin(can.canon(mod2.pack(B)), K8).all())
            total += len(B)
        rep.log(f"u_{k}: done ({total} pairs so far, {hits} hits)")
    rep.check("positive control: x_i u_k + 2L is in K_8 (first block for each k)", ctrl == len(U))
    rep.value("(8,8): pairs tested", total)
    rep.value("(8,8): classes x_i u_k tau + 2L containing a norm-8 vector", hits)
    rep.check("(8,8): none of the 1527984 pairs", total == 1527984 and hits == 0)
    rep.finish()


if __name__ == "__main__":
    main()
