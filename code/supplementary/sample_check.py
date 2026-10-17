"""S4. Route S (sampling; NOT a proof): re-check randomly chosen bounding representatives with
plain int64 arithmetic, without the float32 products and the shared library of verify/
(paper §6.9, Table 13a). The complete checks are verify/v5 and verify/v7.
S4. 経路 S（抜き取り。証明では**ない**）: 無作為に選んだ下界代表元を、verify/ の float32 の積や共有ライブラリを使わず、素朴な int64 の演算で
検算する（論文 §6.9、表 13a）。全数の検算は verify/v5 と verify/v7。

For each sampled bounding representative w0 (odd: of r_j tau + 2L; even: of x_i u_k tau + 2L):
  (a) w0 minus the class representative has even coordinates; (b) its norm is 14 or 18 (odd) or
  12, 16, 20 (even); (c) max |<w0, u>| <= 6 over all minimal vectors u = r s (s in G/{+-1}).
The group is rebuilt here from g1, g2 by breadth-first search.
標本の各下界代表元 w0 について (a) 類の代表との差の座標が偶数、(b) ノルムが 14, 18（奇）または 12, 16, 20（偶）、
(c) すべての最小ベクトル u = r s との |<w0, u>| の最大が 6 以下。群はここで g1, g2 から幅優先で作り直す。

Why it is not a proof: if a fraction p of the entries were wrong, k samples would detect it only
with probability 1 - (1 - p)^k.
証明にならない理由: 誤りが割合 p あっても、k 個の標本で見つかる確率は 1 - (1 - p)^k にすぎない。

Only numpy. Run time: about 10 minutes for k = 200 + 200. Memory: about 1 GB.
numpy のみ。実行時間: k = 200 + 200 で約 10 分。メモリ: 約 1 GB。
Run / 実行:  python3 code/supplementary/sample_check.py [--k 200] [--seed 1]
"""
import argparse
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ints(path):
    return np.array([int(t) for line in open(path) if not line.startswith("#") for t in line.split()], dtype=np.int64)


def npz(name, key):
    with np.load(os.path.join(ROOT, "certificates", name), allow_pickle=False) as z:
        return z[key]


def group_mod_sign(gens, G, order):
    """All elements of <gens> by breadth-first search (int16 storage, int64 products), one of each +-pair.
    <gens> の全元を幅優先で（int16 で保存、int64 で積）。±の片方ずつ。"""
    n = G.shape[0]
    I16 = np.eye(n, dtype=np.int16)
    seen = {I16.tobytes(): I16}
    frontier = [I16]
    while frontier:
        nxt = []
        for A in frontier:
            for g in gens:
                B = A.astype(np.int64) @ g
                assert np.abs(B).max() < 2 ** 15
                B16 = B.astype(np.int16)
                k = B16.tobytes()
                if k not in seen:
                    seen[k] = B16
                    nxt.append(B16)
        frontier = nxt
    els = list(seen.values())
    assert len(els) == order, len(els)
    half = {}
    for g in els:
        half.setdefault(min(g.tobytes(), (-g).tobytes()), g)
    return np.stack(list(half.values())).astype(np.int64)


def main():
    ap = argparse.ArgumentParser(description="S4: sampling check (not a proof)")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    n = 56
    G = ints(os.path.join(ROOT, "data", "L56_gram.txt")).reshape(n, n)
    g1, g2 = ints(os.path.join(ROOT, "data", "sl2_29_generators.txt")).reshape(2, n, n)
    tau = ints(os.path.join(ROOT, "data", "tau.txt")).reshape(n, n)
    E = group_mod_sign([g1, g2], G, 24360)
    print(f"group rebuilt from g1, g2: {len(E)} elements modulo sign", flush=True)
    R6 = npz("min_reps.npz", "reps").astype(np.int64)
    R6T = R6.T.copy()

    def max_ip(w):
        t = E @ (G @ w)                      # (12180, 56): t[s] = s G w^T
        return int(np.abs(t @ R6T).max())    # <r s, w> = r . (s G w^T), all in int64

    W6 = npz("odd_bounding.npz", "w0").astype(np.int64)
    bad = 0
    for j in rng.choice(len(W6), size=min(a.k, len(W6)), replace=False):
        w0 = W6[j]
        bad += not (((w0 - R6[j] @ tau) % 2 == 0).all() and int(w0 @ G @ w0) in (14, 18) and max_ip(w0) <= 6)
    print(f"odd bounding representatives: {min(a.k, len(W6))} sampled, {bad} failed", flush=True)
    bad_odd = bad

    X = npz("norm8_reps.npz", "reps").astype(np.int64)
    U = npz("centralizer.npz", "units").astype(np.int64)
    W8 = npz("even_bounding.npz", "w0")
    bad, hist = 0, {}
    for p in rng.choice(len(W8), size=a.k, replace=False):
        w0 = W8[p].astype(np.int64)
        i, k = divmod(int(p), 6)
        cls = ((w0 - X[i] @ U[k] @ tau) % 2 == 0).all()
        N, m = int(w0 @ G @ w0), max_ip(w0)
        hist[f"{N},{m}"] = hist.get(f"{N},{m}", 0) + 1
        bad += not (cls and N in (12, 16, 20) and m <= 6)
    print(f"even bounding representatives: {a.k} sampled, {bad} failed; (norm, max |ip|) -> count {hist}", flush=True)
    print("detection probability for a fraction p of wrong entries: " +
          ", ".join(f"p = {p:g}: {1 - (1 - p) ** a.k:.3f}" for p in (1e-2, 1e-3, 1e-4, 10 / 1527984)))
    assert bad == 0 and bad_odd == 0


if __name__ == "__main__":
    main()
