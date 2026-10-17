"""V3. F2: L_6 is the disjoint union of the G-orbits of the 656 vectors in certificates/min_reps.npz
(paper §4.4 F2, §6.2, Table 7).
V3. F2: L_6 は certificates/min_reps.npz の 656 本の G-軌道の非交和（論文 §4.4 の F2、§6.2、表 7）。

Two independent arguments are checked / 独立な 2 つの議論を確かめる:

  (A) Expansion. All vectors r s (r a representative, s in G/{+-1}) are computed; up to sign they
      are 7,795,200 distinct vectors of norm 6, i.e. 15,590,400 vectors with both signs. By F1 and
      the theta series (V2), |L_6| = 15,590,400, so they are all of L_6. (Distinctness is shown by
      distinct 64-bit hashes, which implies distinct vectors.)
      展開。r s（r は代表、s ∈ G/{±1}）をすべて計算する。符号を除いて相異なるノルム 6 のベクトルが 7,795,200 本、
      符号込みで 15,590,400 本。F1 とテータ級数（V2）より |L_6| = 15,590,400 なので、これで L_6 全体。
      （相異なることは、64 ビットのハッシュが相異なることで示す。ハッシュが異なればベクトルも異なる。）

  (B) Canonical forms (the argument of the paper). The classes r + 2L have 656 distinct canonical
      forms, so the orbits are distinct. For a vector x of norm 6 the map x -> x + 2L is injective
      up to sign (midpoint lemma 5.1, which uses F1), so |G-orbit of x| = 2 |orbit of x + 2L under
      G/{+-1}|. These sizes add up to 15,590,400 = 632 x 24,360 + 24 x 8,120.
      標準形（論文の議論）。類 r + 2L の標準形は 656 個とも相異なるので、軌道は相異なる。ノルム 6 の x について
      x -> x + 2L は符号を除いて単射（中点補題 5.1。F1 を使う）なので、|x の G-軌道| = 2 |x + 2L の G/{±1}-軌道|。
      この大きさの和が 15,590,400 = 632 × 24,360 + 24 × 8,120。

Exactness: norms in int64; r s in float32 with the exactness bound asserted (r112.shells).
厳密さ: ノルムは int64。r s は float32 で、厳密さの上界を確かめる（r112.shells）。

Run time: about 2 minutes. Memory: about 2 GB.
実行時間: 約 2 分。メモリ: 約 2 GB。

Run / 実行:  python3 code/verify/v3_minimal_orbits.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, mod2, shells  # noqa: E402
from r112.report import Report  # noqa: E402

N6 = 15590400       # |L_6| from the extremal theta series (V2) / 極値テータ級数（V2）から


def main():
    ap = argparse.ArgumentParser(description="V3: F2 (orbits of minimal vectors)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v3_minimal_orbits", a.out)

    G = data.gram(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    rep.log(f"G/{{+-1}}: {len(S)} elements")
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    rep.check("656 representatives", R6.shape == (656, 56))
    rep.check("every representative has norm 6", (exact.norms(R6, G) == 6).all())

    # (A) expansion / 展開
    V = shells.expand_half_shell(R6, S)
    rep.value("vectors r s up to sign (after removing repetitions inside each orbit)", len(V))
    rep.check("2 x 7795200 = 15590400 = |L_6|", 2 * len(V) == N6)
    rep.check("all of them are distinct (distinct 64-bit hashes)", shells.count_distinct_rows(V) == len(V))
    rep.check("all of them have norm 6", (exact.norms(V, G) == 6).all())
    del V

    # (B) canonical forms / 標準形
    can = mod2.Canonical(S)
    can.self_test(S, np.random.default_rng(1))
    forms, sizes = can.canon_and_orbit_size(mod2.pack(R6))
    rep.check("the 656 canonical forms are distinct", len(set(forms.tolist())) == 656)
    vec_sizes = 2 * sizes
    dist = {int(k): int(v) for k, v in zip(*np.unique(vec_sizes, return_counts=True))}
    rep.value("orbit sizes {size: number of orbits}", dist)
    rep.check("sizes add up to 15590400 = 632 x 24360 + 24 x 8120",
              int(vec_sizes.sum()) == N6 and dist == {8120: 24, 24360: 632})
    rep.finish()


if __name__ == "__main__":
    main()
