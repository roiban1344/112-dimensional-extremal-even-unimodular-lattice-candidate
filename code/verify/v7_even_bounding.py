"""V7. F5b by even bounding representatives: min(x_i u_k tau + 2L) >= 12 for all 1,527,984 pairs
(paper §4.4 F5b, Corollary 5.3, Definition 5.4, §6.7, Table 12).
V7. 偶の下界代表元による F5b: 1,527,984 組すべてについて min(x_i u_k tau + 2L) ≥ 12
（論文 §4.4 の F5b、系 5.3、定義 5.4、§6.7、表 12）。

Row 6 i + k of certificates/even_bounding.npz is a vector w0 with
certificates/even_bounding.npz の第 6i+k 行はベクトル w0 で、
  (a) w0 - x_i u_k tau has even coordinates (w0 lies in the class);
      w0 - x_i u_k tau の座標がすべて偶数（w0 は類に入る）;
  (b) <w0, w0> is 12, 16 or 20;
      <w0, w0> は 12, 16, 20 のどれか;
  (c) |<w0, u>| <= 6 for every minimal vector u of L.
      L のすべての最小ベクトル u について |<w0, u>| ≤ 6。
By Corollary 5.3 the class contains no vector of norm 8. Together with F3 (V4) this proves
condition (ii) of Corollary 4.3 without canonical forms (an alternative to V6).
系 5.3 より、類はノルム 8 のベクトルを含まない。F3（V4）と合わせて、標準形を使わずに系 4.3 の条件 (ii) が示される（V6 の代わり）。

(c) is the expensive part: 1.5 million x 7.8 million inner products. It is done in chunks with
a checkpoint (out/v7_even_bounding.state.npz), so an interrupted run resumes with the same
command. Use --threads to run several BLAS-free Python threads (each matrix product is
single-threaded then).
(c) が重い部分で、150 万 × 780 万の内積。チャンクごとに途中経過（out/v7_even_bounding.state.npz）を保存するので、
止まっても同じコマンドで再開する。--threads で複数の Python のスレッドを使う（そのとき各行列積は 1 スレッド）。

Exactness: (a), (b) in int64; (c) in float32 with the exactness bound asserted for every chunk.
厳密さ: (a), (b) は int64。(c) は float32 で、チャンクごとに厳密さの上界を確かめる。

Run time: about 8 CPU hours for (c) (e.g. 3 hours with --threads 3). Memory: about 3 GB.
実行時間: (c) に約 8 CPU 時間（--threads 3 なら約 3 時間）。メモリ: 約 3 GB。

Run / 実行:  python3 code/verify/v7_even_bounding.py [--threads T] [--limit N] [--data DIR] [--certificates DIR] [--out DIR]
             (--limit N checks only the first N pairs, for a quick test / 先頭の N 組だけを確かめる。試験用)
"""
import os
import sys

if "--threads" in sys.argv:
    # One BLAS thread per Python thread. Must be set before numpy is imported.
    # Python のスレッド 1 本につき BLAS も 1 スレッド。numpy を読み込む前に設定する。
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import argparse  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, shells  # noqa: E402
from r112.report import Report  # noqa: E402

NPAIRS = 1527984


def main():
    ap = argparse.ArgumentParser(description="V7: F5b by even bounding representatives")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=8192, help="pairs per checkpoint / 途中経過の保存ごとの組数")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v7_even_bounding" + (f"_first{a.limit}" if a.limit else ""), a.out)

    G = data.gram(a.data)
    t = data.tau(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data))
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
    U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    W = certs.load("even_bounding.npz", "w0", a.certificates)
    rep.check("1527984 bounding representatives, one per pair (row 6 i + k)", W.shape == (NPAIRS, 56) and 6 * len(X) == NPAIRS)
    n = a.limit or NPAIRS
    mm = exact.i64_matmul

    # (a), (b): grouped by k, rows k, k + 6, k + 12, ... / k ごとにまとめる（第 k, k+6, k+12, … 行）
    ok_a = True
    norms = np.empty(NPAIRS, dtype=np.int64)
    for k, u in enumerate(U):
        Wk = W[k::6].astype(np.int64)
        start = mm(mm(X, u), t)
        ok_a &= bool(((Wk - start) % 2 == 0).all())
        norms[k::6] = exact.norms(Wk, G)
    rep.check("(a) w0 - x_i u_k tau has even coordinates, for all pairs", ok_a)
    hist_norm = {int(v): int(c) for v, c in zip(*np.unique(norms, return_counts=True))}
    rep.value("(b) norms {norm: count}", hist_norm)
    rep.check("(b) every <w0, w0> is 12, 16 or 20", np.isin(norms, (12, 16, 20)).all())

    # (c) with checkpoints / 途中経過を保存しながら
    mv = shells.MinimalVectors(G, R6, S)
    rep.log(f"expanded {2 * mv.count} minimal vectors; checking (c) for {n} pairs with {a.threads} thread(s)")
    state = os.path.join(rep.out_dir, rep.step + ".state.npz")
    if os.path.exists(state):
        with np.load(state, allow_pickle=False) as z:
            mx, done, secs = z["mx"].copy(), int(z["done"]), float(z["seconds"])
        assert len(mx) == n
        rep.log(f"resuming at {done} / {n}")
    else:
        mx, done, secs = np.full(n, -1, dtype=np.int8), 0, 0.0
    last = time.time()
    while done < n:
        t1 = time.time()
        e = min(n, done + a.chunk)
        mx[done:e] = mv.max_abs_inner(W[done:e], threads=a.threads)
        secs += time.time() - t1
        done = e
        if time.time() - last > 300 or done == n:
            os.makedirs(rep.out_dir, exist_ok=True)
            certs.save_npz(state, mx=mx, done=np.int64(done), seconds=np.float64(secs))
            last = time.time()
            rep.log(f"(c) {done} / {n}: {1000 * secs / done:.1f} ms per pair, max so far {int(mx[:done].max())}")
    table = {}
    for (nn, m), c in zip(*np.unique(np.stack([norms[:n], mx.astype(np.int64)], axis=1), axis=0, return_counts=True)):
        table[f"norm {nn}, max {m}"] = int(c)
    rep.value("Table 12: (norm, max |<w0, u>|) -> count", table)
    rep.value("seconds spent in (c)", round(secs))
    rep.check(f"(c) max |<w0, u>| <= 6 over all minimal vectors u, for all {n} pairs", (mx <= 6).all() and (mx >= 0).all())
    if n < NPAIRS:
        rep.log(f"NOTE: only the first {n} pairs were checked (--limit)")
    rep.finish()


if __name__ == "__main__":
    main()
