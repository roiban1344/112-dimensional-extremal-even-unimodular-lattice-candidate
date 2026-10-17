"""V5. F5a: min(r_j tau + 2L) >= 14 for the 656 representatives r_j of F2, by odd bounding
representatives (paper §4.4 F5a, Corollary 5.2, Definition 5.4, §6.6, Table 11).
V5. F5a: F2 の 656 個の代表 r_j について min(r_j tau + 2L) ≥ 14。奇の下界代表元による
（論文 §4.4 の F5a、系 5.2、定義 5.4、§6.6、表 11）。

For each j, row j of certificates/odd_bounding.npz is a vector w0 with
各 j について、certificates/odd_bounding.npz の第 j 行はベクトル w0 で、
  (a) w0 - r_j tau has even coordinates (w0 lies in the class r_j tau + 2L);
      w0 - r_j tau の座標がすべて偶数（w0 は類 r_j tau + 2L に入る）;
  (b) <w0, w0> is 14 or 18;
      <w0, w0> は 14 か 18;
  (c) |<w0, u>| <= 6 for every minimal vector u of L (all 15,590,400, expanded from F2).
      L のすべての最小ベクトル u（F2 から展開した 15,590,400 本）について |<w0, u>| ≤ 6。
By Corollary 5.2 the class then contains no vector of norm 6 or 10, i.e. its minimum is >= 14.
This settles the collision types (6,6) and (6,10); the type (10,6) is the same as (6,10)
because tau^2 = -5 is the identity modulo 2 (Appendix C.4).
系 5.2 より、類はノルム 6 も 10 も含まない。つまり最小は 14 以上。これで衝突の型 (6,6) と (6,10) が片づく。
型 (10,6) は (6,10) と同じ（tau^2 = -5 は 2 を法として恒等写像なので。付録 C.4）。

Exactness: (a), (b) in int64; (c) in float32 with the exactness bound asserted (r112.shells).
厳密さ: (a), (b) は int64。(c) は float32 で、厳密さの上界を確かめる（r112.shells）。

Run time: about 2 minutes. Memory: about 2 GB.
実行時間: 約 2 分。メモリ: 約 2 GB。

Run / 実行:  python3 code/verify/v5_odd_bounding.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, shells  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="V5: F5a (odd bounding representatives)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v5_odd_bounding", a.out)

    G = data.gram(a.data)
    t = data.tau(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    W = certs.load("odd_bounding.npz", "w0", a.certificates).astype(np.int64)
    rep.check("one bounding representative per representative r_j (656)", W.shape == R6.shape == (656, 56))

    rep.check("tau^2 = -I modulo 2 (so the types (6,10) and (10,6) coincide)",
              ((exact.i64_matmul(t, t) + np.eye(56, dtype=np.int64)) % 2 == 0).all())
    rep.check("(a) w0 - r_j tau has even coordinates", ((W - exact.i64_matmul(R6, t)) % 2 == 0).all())
    norms = exact.norms(W, G)
    rep.value("(b) norms {norm: count}", {int(k): int(v) for k, v in zip(*np.unique(norms, return_counts=True))})
    rep.check("(b) every <w0, w0> is 14 or 18", np.isin(norms, (14, 18)).all())

    mv = shells.MinimalVectors(G, R6, S)
    rep.log(f"expanded {2 * mv.count} minimal vectors")
    m = mv.max_abs_inner(W)
    table = {}
    for nn, mx in zip(norms.tolist(), m.tolist()):
        table[f"norm {nn}, max {mx}"] = table.get(f"norm {nn}, max {mx}", 0) + 1
    rep.value("Table 11: (norm, max |<w0, u>|) -> count", dict(sorted(table.items())))
    rep.check("(c) max |<w0, u>| <= 6 over all minimal vectors u, for all 656", (m <= 6).all())
    rep.finish()


if __name__ == "__main__":
    main()
