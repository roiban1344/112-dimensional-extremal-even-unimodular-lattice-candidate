"""P2. Find representatives of the G-orbits on L_6 by a closure (paper §6.2). Produces the list
that verify/v3_minimal_orbits.py checks; how it is found does not matter for the proof.
P2. 閉包によって L_6 の G-軌道の代表を見つける（論文 §6.2）。verify/v3_minimal_orbits.py が確かめる一覧を作る。
作り方は証明に関係しない。

Closure step: if r, s are minimal vectors with <r, s> = 3, then r - s is again minimal
(<r - s, r - s> = 6 + 6 - 6 = 6). Starting from the minimal vectors in the reduced basis
(certificates/reduction_L.txt), for each new orbit representative x we look at the vectors y of
some known orbits with <x, y> = 3; whenever x - y lies in an orbit not yet known, its orbit is
added. The search stops when the orbit sizes add up to |L_6| = 15,590,400. (A version that moved
only by group elements would never leave one orbit; the closure step is what finds new ones.)
閉包の一歩: r, s が最小ベクトルで <r, s> = 3 なら r - s も最小（<r-s, r-s> = 6 + 6 - 6 = 6）。簡約基底（certificates/reduction_L.txt）
の中の最小ベクトルから始め、新しい軌道の代表 x ごとに、既知のいくつかの軌道の中で <x, y> = 3 となる y を見る。x - y が未知の軌道に
入っていれば、その軌道を足す。軌道の大きさの和が |L_6| = 15,590,400 に達したら止める。（群の元で動かすだけでは 1 つの軌道から出られない。
新しい軌道を見つけるのは閉包の一歩。）

Known vectors are stored as 64-bit hashes (both signs, 15.6 million at the end). A hash collision
could only make the search skip an orbit, which would show up as a missing orbit (the size sum
would not be reached, or V3 would fail); the final list is checked independently by V3.
既知のベクトルは 64 ビットのハッシュで持つ（符号込み、最後は 1,560 万個）。ハッシュの衝突は軌道を飛ばす方向にしか働かず、
それは軌道の欠けとして現れる（大きさの和に届かないか、V3 が失敗する）。最終の一覧は V3 が独立に確かめる。

Output: out/p2_min_reps.npz (reps, int16 (656, 56)). Compared with certificates/min_reps.npz by
the set of canonical forms (the order and the choice of representatives may differ).
出力: out/p2_min_reps.npz。certificates/min_reps.npz とは標準形の集合で比べる（並びや代表の選び方は違いうる）。

Deterministic (fixed random seed). Run time: about 5 minutes. Memory: about 1.5 GB.
決定的（乱数の種は固定）。実行時間: 約 5 分。メモリ: 約 1.5 GB。

Run / 実行:  python3 code/produce/p2_minimal_orbits.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, mod2  # noqa: E402
from r112.report import Report  # noqa: E402

N6 = 15590400


def minimal_orbits(G, S, seeds, rep, n6=N6):
    """Representatives of the G-orbits on the minimal vectors (norm 6) of the lattice with Gram matrix G,
    by the closure described above, starting from `seeds`; stops when the orbit sizes reach n6.
    Usable for any lattice with an action of SL_2(29) (also Nebe's L_{56,2}(M) in supplementary/nebe).
    Gram 行列 G の格子の最小ベクトル（ノルム 6）の G-軌道の代表を、上の閉包で `seeds` から求める。軌道の大きさの和が n6 に達したら止める。
    SL_2(29) が作用する任意の格子に使える（supplementary/nebe の Nebe の L_{56,2}(M) にも）。"""
    n = len(G)
    S32 = S.astype(np.float32)
    smax = int(np.abs(S).max())
    rng = random.Random(0)
    hv = np.array([rng.getrandbits(62) | 1 for _ in range(n)], dtype=np.int64)   # hash weights / ハッシュの重み

    def orbit(x):
        """The G-orbit of x with both signs, as distinct int64 rows. / x の G-軌道（符号込み、相異なる行）。"""
        assert int(np.abs(x).sum()) * smax < 2 ** 24                               # float32 exactness / float32 の厳密さ
        O = np.rint(x.astype(np.float32) @ S32).astype(np.int64)
        return np.unique(np.concatenate([O, -O]), axis=0)

    reps, queue = [], []
    known = np.zeros(0, dtype=np.int64)          # sorted hashes of all known vectors / 既知のベクトルのハッシュ（整列済み）
    total = 0

    def is_known(v):
        h = np.int64(v @ hv)
        pos = min(int(np.searchsorted(known, h)), len(known) - 1)
        return len(known) > 0 and known[pos] == h

    def add(x):
        nonlocal known, total
        O = orbit(x)
        assert (exact.norms(O[:1], G) == 6).all()
        reps.append(x)
        queue.append(len(reps) - 1)
        total += len(O)
        known = np.sort(np.concatenate([known, O @ hv]))

    for s in seeds:
        if total >= n6:
            break
        if not is_known(s):
            add(np.asarray(s, dtype=np.int64))
    while total < n6 and queue:
        i = queue.pop(0)
        x = reps[i]
        Gx = G @ x
        for j in rng.sample(range(len(reps)), min(len(reps), 12)) + [i]:
            Oy = orbit(reps[j])                                                  # recomputed: storing all would need ~7 GB / 全部を持つと約 7 GB 要るので作り直す
            cand = x[None, :] - Oy[(Oy @ Gx) == 3]                              # x - y, <x, y> = 3
            while len(cand) and total < n6:
                hc = cand @ hv
                pos = np.searchsorted(known, hc)
                pos[pos == len(known)] = 0
                new = cand[known[pos] != hc]
                if len(new) == 0:
                    break
                add(new[0])
                cand = new[1:]
            if total >= n6:
                break
        if len(reps) % 100 == 0:
            rep.log(f"{len(reps)} orbits, {total} vectors")
    rep.value("orbits found", len(reps))
    rep.value("vectors covered", total)
    rep.check("the orbit sizes add up to |L_6|", total == n6)
    rep.check("no hash collision among the known vectors", len(np.unique(known)) == len(known))
    return np.stack(reps)


def main():
    ap = argparse.ArgumentParser(description="P2: orbits of minimal vectors by closure")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("p2_minimal_orbits", a.out)
    G = data.gram(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    # Seeds: norm-6 rows of the reduced basis and their sums/differences of norm 6.
    # 種: 簡約基底のノルム 6 の行と、その和・差でノルム 6 のもの。
    U = certs.load_matrix("reduction_L.txt", 56, a.certificates)
    B6 = [u for u in U if int(u @ G @ u) == 6]
    seeds = B6 + [x + s * y for i, x in enumerate(B6) for y in B6[:i] for s in (1, -1) if int((x + s * y) @ G @ (x + s * y)) == 6]
    rep.value("seed minimal vectors", len(seeds))

    R = minimal_orbits(G, S, seeds, rep)
    os.makedirs(rep.out_dir, exist_ok=True)
    certs.save_npz(os.path.join(rep.out_dir, "p2_min_reps.npz"), reps=R.astype(np.int16))

    # Compare with the certificate by canonical forms. / 配布した代表と標準形で比べる。
    can = mod2.Canonical(S)
    mine = set(can.canon(mod2.pack(R)).tolist())
    old = set(can.canon(mod2.pack(certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64))).tolist())
    rep.check("same 656 orbits as certificates/min_reps.npz (equal sets of canonical forms)", mine == old and len(mine) == 656)
    rep.finish()


if __name__ == "__main__":
    main()
