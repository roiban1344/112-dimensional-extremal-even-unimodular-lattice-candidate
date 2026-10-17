"""V4. F3: L_8 is the union of the G-orbits of the 1,527,984 vectors x_i u_k, and consists of
1,517,130 regular G-orbits (paper §4.4 F3, §6.4, Table 7).
V4. F3: L_8 は 1,527,984 本のベクトル x_i u_k の G-軌道の和集合で、1,517,130 個の正則な G-軌道からなる
（論文 §4.4 の F3、§6.4、表 7）。

Data: x_i (i < 254,664) from certificates/norm8_reps.npz and u_k (k < 6) from
certificates/centralizer.npz. The u_k are only a convenient way to list vectors here: the
argument below uses nothing about them except that the products x_i u_k are vectors of L.
データ: x_i（i < 254,664）は certificates/norm8_reps.npz、u_k（k < 6）は certificates/centralizer.npz。
ここでは u_k はベクトルを並べる手段にすぎない。下の議論が使うのは、積 x_i u_k が L のベクトルであることだけ。

Argument (as in V3 (B)) / 議論（V3 (B) と同じ）:
  * every x_i u_k has norm 8;
    x_i u_k はすべてノルム 8;
  * the classes x_i u_k + 2L have exactly 1,517,130 distinct canonical forms;
    類 x_i u_k + 2L の標準形はちょうど 1,517,130 種類;
  * for norm-8 vectors x -> x + 2L is injective up to sign (midpoint lemma 5.1, uses F1), so
    distinct canonical forms are distinct G-orbits and |G-orbit of x| = 2 |orbit of x + 2L|;
    ノルム 8 のベクトルでは x -> x + 2L は符号を除いて単射（中点補題 5.1。F1 を使う）なので、
    標準形が異なれば G-軌道も異なり、|x の G-軌道| = 2 |x + 2L の軌道|;
  * over the distinct canonical forms these sizes add up to 36,957,286,800 = |L_8| (V2), and all
    orbits have the full size 24,360. Hence the listed orbits are all of L_8.
    相異なる標準形にわたる大きさの和が 36,957,286,800 = |L_8|（V2）で、どの軌道も最大の大きさ 24,360。
    よって並べた軌道で L_8 全体。
The table K_8 of these canonical forms is written to out/K8.npy for V6, and one index 6 i + k per
orbit to out/K8_first.npy for V8.
この標準形の表 K_8 を V6 のために out/K8.npy に、軌道ごとの添字 6i+k を 1 つずつ V8 のために out/K8_first.npy に書く。

Also checked (not needed for F3): each u_k is an isometry of L commuting with g1, g2, and the
u_k are distinct modulo +-1.
あわせて確かめる（F3 には不要）: 各 u_k は g1, g2 と可換な L の等長変換で、±1 を法として相異なる。

Exactness: integer arithmetic only (int64 with overflow guards, uint64 XOR).
厳密さ: 整数演算のみ（オーバーフローを確かめた int64、uint64 の XOR）。

Run time: about 10 minutes. Memory: about 1.5 GB.
実行時間: 約 10 分。メモリ: 約 1.5 GB。

Run / 実行:  python3 code/verify/v4_norm8_orbits.py [--data DIR] [--certificates DIR] [--out DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, mod2  # noqa: E402
from r112.report import Report  # noqa: E402

N8 = 36957286800    # |L_8| from the extremal theta series (V2) / 極値テータ級数（V2）から


def main():
    ap = argparse.ArgumentParser(description="V4: F3 (orbits of norm-8 vectors)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("v4_norm8_orbits", a.out)

    G = data.gram(a.data)
    gens = data.generators(a.data)
    S = group.sl2_29_mod_sign(gens)
    X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
    U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    rep.check("254664 representatives x_i and 6 matrices u_k", X.shape == (254664, 56) and U.shape == (6, 56, 56))

    # Facts about u_k (not needed for F3). / u_k についての事実（F3 には不要）。
    mm = exact.i64_matmul
    rep.check("each u_k is an isometry of L (u G u^T = G)", all((mm(mm(u, G), u.T) == G).all() for u in U))
    rep.check("each u_k commutes with g1 and g2", all((mm(u, g) == mm(g, u)).all() for u in U for g in gens))
    keys = {group.sign_normalize(u).tobytes() for u in U}
    rep.check("the u_k are distinct modulo +-1", len(keys) == 6)

    # The 1,527,984 vectors x_i u_k, in the order 6 i + k. / 6i+k の順に並べた x_i u_k。
    Y = np.stack([mm(X, u) for u in U], axis=1).reshape(-1, 56)
    rep.check("1527984 vectors x_i u_k", len(Y) == 1527984)
    rep.check("every x_i u_k has norm 8", (exact.norms(Y, G) == 8).all())

    can = mod2.Canonical(S)
    can.self_test(S, np.random.default_rng(2))
    forms = np.empty(len(Y), dtype=np.uint64)
    sizes = np.empty(len(Y), dtype=np.int64)
    step = 1 << 15
    for i in range(0, len(Y), step):
        forms[i:i + step], sizes[i:i + step] = can.canon_and_orbit_size(mod2.pack(Y[i:i + step]))
        if (i // step) % 8 == 0:
            rep.log(f"canonical forms: {min(i + step, len(Y))} / {len(Y)}")
    K8, first = np.unique(forms, return_index=True)
    rep.value("distinct canonical forms (= distinct G-orbits)", len(K8))
    rep.check("exactly 1517130 distinct G-orbits", len(K8) == 1517130)
    vec_sizes = 2 * sizes[first]
    rep.value("orbit sizes {size: number of orbits}", {int(k): int(v) for k, v in zip(*np.unique(vec_sizes, return_counts=True))})
    rep.check("all orbits are regular (size 24360)", (vec_sizes == 24360).all())
    rep.check("sizes add up to |L_8| = 36957286800", int(vec_sizes.sum()) == N8)
    os.makedirs(rep.out_dir, exist_ok=True)
    np.save(os.path.join(rep.out_dir, "K8.npy"), K8)
    np.save(os.path.join(rep.out_dir, "K8_first.npy"), first)
    rep.log("wrote out/K8.npy (sorted canonical forms of the norm-8 classes) and out/K8_first.npy "
            "(for each, the first index 6 i + k with that canonical form)")
    rep.finish()


if __name__ == "__main__":
    main()
