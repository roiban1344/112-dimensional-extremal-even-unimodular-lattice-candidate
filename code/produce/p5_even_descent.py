"""P5. Find the even bounding representatives of the 1,527,984 classes x_i u_k tau + 2L
(paper §6.7, Table 12). Produces certificates/even_bounding.npz (checked by verify/v7_even_bounding.py).
P5. 1,527,984 個の類 x_i u_k tau + 2L の偶の下界代表元を見つける（論文 §6.7、表 12）。certificates/even_bounding.npz を作る
（verify/v7_even_bounding.py が確かめる）。

For pair number p = 6 i + k, run the descent (descent.py) from w = x_i u_k tau (norm 40).
Expected: norm 12, 16 or 20 and maximal inner product 5 or 6 for every pair.
組の番号 p = 6i + k について、w = x_i u_k tau（ノルム 40）から降下（descent.py）を走らせる。期待: どの組もノルム 12, 16, 20 で内積の最大は 5 か 6。

The work is split into `nparts` interleaved parts (part q takes the pairs p = q, q + nparts, ...),
each saved every 5,000 pairs to out/p5/part<q>of<nparts>.npz and resumable with the same command.
The parts are then merged with --merge into out/p5_even_bounding.npz.
仕事を `nparts` 個の飛び飛びの部分に分ける（部分 q は p = q, q + nparts, … の組）。各部分は 5,000 組ごとに out/p5/part<q>of<nparts>.npz に保存し、
同じコマンドで再開できる。最後に --merge で out/p5_even_bounding.npz にまとめる。

Run time: the certificates were produced in 6 parts of about 31 hours each on a shared machine
(about 190 CPU hours, 0.4-0.7 s per pair). Use --limit for a test: the first pairs of the part are
compared with certificates/even_bounding.npz.
実行時間: 配布した代表元は、共用の計算機で約 31 時間ずつの 6 部分で作った（約 190 CPU 時間、1 組 0.4〜0.7 秒）。試すときは --limit を使う。
その部分の先頭の組を certificates/even_bounding.npz と比べる。

Run / 実行:  python3 code/produce/p5_even_descent.py --part Q --nparts P [--limit N]
             python3 code/produce/p5_even_descent.py --merge --nparts P
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from descent import Descent  # noqa: E402
from r112 import certs, data, exact, group  # noqa: E402
from r112.report import Report  # noqa: E402

NPAIRS = 1527984


def main():
    ap = argparse.ArgumentParser(description="P5: even bounding representatives")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs of this part / この部分の先頭 N 組だけ")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report(f"p5_even_descent_part{a.part}of{a.nparts}" if not a.merge else "p5_even_descent_merge", a.out)
    pdir = os.path.join(rep.out_dir, "p5")
    os.makedirs(pdir, exist_ok=True)

    if a.merge:
        W = np.zeros((NPAIRS, 56), dtype=np.int16)
        for q in range(a.nparts):
            with np.load(os.path.join(pdir, f"part{q}of{a.nparts}.npz"), allow_pickle=False) as z:
                w, done = z["w0"], int(z["done"])
            idx = np.arange(q, NPAIRS, a.nparts)
            rep.check(f"part {q} is complete", done == len(idx) == len(w))
            W[idx] = w
        certs.save_npz(os.path.join(rep.out_dir, "p5_even_bounding.npz"), w0=W)
        old = certs.load("even_bounding.npz", "w0", a.certificates)
        rep.check("identical to certificates/even_bounding.npz", (old == W).all())
        rep.finish()
        return

    G = data.gram(a.data)
    t = data.tau(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
    U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    d = Descent(G, S, R6)

    idx = np.arange(a.part, NPAIRS, a.nparts)
    if a.limit:
        idx = idx[:a.limit]
    path = os.path.join(pdir, f"part{a.part}of{a.nparts}" + (f"-first{a.limit}" if a.limit else "") + ".npz")
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            W, done = z["w0"].copy(), int(z["done"])
        rep.log(f"resuming at {done} / {len(idx)}")
    else:
        W, done = np.zeros((len(idx), 56), dtype=np.int16), 0
    hist = {}
    t0 = time.time()
    for pos in range(done, len(idx)):
        p = int(idx[pos])
        i, k = divmod(p, 6)
        w0 = exact.i64_matmul(exact.i64_matmul(X[i], U[k]), t)
        w, m, _ = d.descend(w0, 80)
        assert ((w - w0) % 2 == 0).all() and np.abs(w).max() < 2 ** 15
        nw = int(w @ G @ w)
        key = f"norm {nw}, max {m}"
        hist[key] = hist.get(key, 0) + 1
        if not (nw in (12, 16, 20) and m <= 6):
            rep.log(f"pair {p}: no bounding representative found (norm {nw}, max {m})")
        W[pos] = w
        if (pos + 1) % 5000 == 0 or pos + 1 == len(idx):
            certs.save_npz(path, w0=W, done=np.int64(pos + 1))
            rep.log(f"{pos + 1} / {len(idx)} pairs, {(time.time() - t0) / (pos + 1 - done):.2f} s per pair, {hist}")
    rep.value("(norm, max |<w0, u>|) -> count (this run)", hist)
    if a.limit:
        old = certs.load("even_bounding.npz", "w0", a.certificates)
        rep.check(f"the first {a.limit} pairs of part {a.part} are identical to certificates/even_bounding.npz",
                  (old[idx] == W).all())
    rep.finish()


if __name__ == "__main__":
    main()
