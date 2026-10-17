"""P4. Find the 656 odd bounding representatives of the classes r_j tau + 2L (paper §6.6, Table 11).
Produces certificates/odd_bounding.npz (checked by verify/v5_odd_bounding.py).
P4. 類 r_j tau + 2L の奇の下界代表元 656 個を見つける（論文 §6.6、表 11）。certificates/odd_bounding.npz を作る
（verify/v5_odd_bounding.py が確かめる）。

For each representative r_j (certificates/min_reps.npz), run the descent (descent.py) from
w = r_j tau (norm 30). Expected result: norm 14 (243 classes) or 18 (413 classes), maximal inner
product 5 or 6.
各代表 r_j について、w = r_j tau（ノルム 30）から降下（descent.py）を走らせる。期待される結果: ノルム 14（243 類）か 18（413 類）、
内積の最大は 5 か 6。

Positive control (as in the paper): starting from (a vector of norm 10) + (an element of 2L), the
descent must end at norm 10 and never output a false bounding representative. The norm-10
vectors are made as r + s or r - s with minimal r, s and <r, s> = -1 or 1.
陽性対照（論文と同じ）: (ノルム 10 のベクトル) + (2L の元) から始めると、降下はノルム 10 で止まり、偽の下界代表元を出さないはず。
ノルム 10 のベクトルは、最小ベクトル r, s（<r, s> = ∓1）の r ± s として作る。

Output: out/p4_odd_bounding.npz (w0, int16 (656, 56)); compared with certificates/odd_bounding.npz.
出力: out/p4_odd_bounding.npz。certificates/odd_bounding.npz と比べる。

Run time: about 3 minutes. Memory: about 1 GB.
実行時間: 約 3 分。メモリ: 約 1 GB。

Run / 実行:  python3 code/produce/p4_odd_descent.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from descent import Descent  # noqa: E402
from r112 import certs, data, exact, group  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="P4: odd bounding representatives")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("p4_odd_descent", a.out)

    G = data.gram(a.data)
    t = data.tau(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    d = Descent(G, S, R6)

    # Positive control. / 陽性対照。
    rng = random.Random(1)
    vec = lambda: R6[rng.randrange(len(R6))] @ S[rng.randrange(len(S))].astype(np.int64)   # noqa: E731
    returned = false_cert = tried = 0
    while tried < 60:
        r, s = vec(), vec()
        ip = int(r @ G @ s)
        if abs(ip) != 1:
            continue
        w10 = r - s if ip == 1 else r + s
        assert int(w10 @ G @ w10) == 10
        shift = sum(rng.choice((-1, 1)) * vec() for _ in range(3))
        w, m, _ = d.descend(w10 + 2 * shift, 40)
        nw = int(w @ G @ w)
        tried += 1
        returned += nw == 10
        false_cert += (nw in (14, 18) and m <= 6)
    rep.value("positive control: planted norm-10 classes, descent ended at norm 10", f"{returned} of {tried}")
    rep.check("positive control: no false bounding representative", false_cert == 0)

    W = np.zeros((len(R6), 56), dtype=np.int64)
    table = {}
    for j, r in enumerate(R6):
        w0 = exact.i64_matmul(r, t)
        w, m, _ = d.descend(w0, 40)
        assert ((w - w0) % 2 == 0).all()
        nw = int(w @ G @ w)
        key = f"norm {nw}, max {m}"
        table[key] = table.get(key, 0) + 1
        W[j] = w
    rep.value("Table 11: (norm, max |<w0, u>|) -> count", dict(sorted(table.items())))
    ok = all(k.split(",")[0] in ("norm 14", "norm 18") and int(k.split()[-1]) <= 6 for k in table)
    rep.check("every class got a bounding representative (norm 14 or 18, max <= 6)", ok)
    os.makedirs(rep.out_dir, exist_ok=True)
    certs.save_npz(os.path.join(rep.out_dir, "p4_odd_bounding.npz"), w0=W.astype(np.int16))
    old = certs.load("odd_bounding.npz", "w0", a.certificates).astype(np.int64)
    rep.check("identical to certificates/odd_bounding.npz", (old == W).all())
    rep.finish()


if __name__ == "__main__":
    main()
