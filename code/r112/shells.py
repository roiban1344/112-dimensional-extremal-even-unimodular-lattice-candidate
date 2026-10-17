"""The minimal vectors of L and inner products with them (paper §5, Definition 5.4 (c)).
L の最小ベクトルと、それとの内積（論文 §5、定義 5.4 (c)）。

The 15,590,400 minimal vectors are the G-orbits of the 656 representatives of F2.
Since -1 is in G, it suffices to keep one of each pair +-u: the set
    V = { sign-normalized r_j s : j < 656, s in G/{+-1} }
has 7,795,200 elements, which is asserted when it is built. For a bounding representative
we need max_{u in L_6} |<w, u>| = max_{v in V} |<w, v>|.
15,590,400 本の最小ベクトルは、F2 の 656 個の代表の G-軌道。-1 ∈ G なので ±u の片方だけ持てばよい。
    V = { r_j s の符号を正規化したもの : j < 656, s ∈ G/{±1} }
は 7,795,200 個で、作るときにこの個数を確かめる。下界代表元に必要なのは max_{u ∈ L_6} |<w, u>| = max_{v ∈ V} |<w, v>|。

Exactness: the products V @ (G W^T) are computed in float32 in blocks; for every block
the bound of r112.exact (all partial sums < 2^24) is asserted, so the results are exact.
厳密さ: V @ (G W^T) はブロックごとに float32 で計算する。各ブロックで r112.exact の上界（部分和がすべて 2^24 未満）を確かめるので、結果は厳密。
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import exact

N_MIN = 15590400


def expand_half_shell(reps, S):
    """V = one of each pair +-u of the vectors r s (r in reps, s in S), as int16 (k, 56).
    Duplicates inside one orbit (from a nontrivial stabilizer) are removed; the caller
    checks that different orbits do not overlap (see `MinimalVectors`).
    r s（r ∈ reps, s ∈ S）の ±u の片方ずつ（int16 (k, 56)）。1 つの軌道の中の重複（固定群が自明でないときに生じる）は除く。
    異なる軌道が重ならないことは呼び出し側で確かめる（`MinimalVectors`）。"""
    S = np.asarray(S)
    S32 = S.astype(np.float32)
    smax = int(np.abs(S).max())
    parts = []
    for r in np.asarray(reps, dtype=np.int64):
        # r s in float32 is exact if sum_j |r_j| max|s| < 2^24. / 厳密さの条件を確かめる。
        assert int(np.abs(r).sum()) * smax < 2 ** 24
        Y = np.rint(r.astype(np.float32) @ S32).astype(np.int64)            # (|S|, 56)
        # Normalize the sign: first nonzero coordinate positive. / 符号の正規化: 最初の 0 でない座標を正に。
        first = (Y != 0).argmax(axis=1)
        sgn = np.sign(Y[np.arange(len(Y)), first])
        assert (sgn != 0).all()
        parts.append(np.unique((Y * sgn[:, None]).astype(np.int16), axis=0))
    return np.concatenate(parts)


def count_distinct_rows(V, seed=20261017):
    """Number of distinct 64-bit hashes of the rows of V. Equal rows have equal hashes, so
    this is at most the number of distinct rows; if it equals len(V), all rows are distinct.
    V の行の 64 ビットのハッシュの種類数。等しい行は等しいハッシュをもつので、これは相異なる行の数以下。
    len(V) に等しければ、すべての行は相異なる。"""
    rng = np.random.default_rng(seed)
    h = rng.integers(1, 2 ** 62, size=V.shape[1], dtype=np.int64) | 1
    # int64 matrix products wrap around modulo 2^64; that is fine for hashing. Blocks keep memory small.
    # int64 の行列積は 2^64 を法として回り込むが、ハッシュとしては問題ない。ブロックに分けてメモリを抑える。
    keys = np.concatenate([V[i:i + (1 << 20)].astype(np.int64) @ h for i in range(0, len(V), 1 << 20)])
    return len(np.unique(keys))


class MinimalVectors:
    """All minimal vectors of L (one of each +-pair) and maximal inner products with them.
    L のすべての最小ベクトル（±の片方ずつ）と、それとの内積の最大。"""

    def __init__(self, G, reps, S):
        self.G = np.asarray(G, dtype=np.int64)
        self.V = expand_half_shell(reps, S)
        assert 2 * len(self.V) == N_MIN, f"{2 * len(self.V)} minimal vectors, expected {N_MIN}"
        assert count_distinct_rows(self.V) == len(self.V), "two orbits overlap"
        # Every expanded vector has norm 6 (checked exactly in int64).
        # 展開したベクトルがすべてノルム 6 であることを int64 で厳密に確かめる。
        assert (exact.norms(self.V, self.G) == 6).all()
        self.count = len(self.V)
        self.vmax = int(np.abs(self.V).max())

    def max_abs_inner(self, W, threads=1, block_w=512, block_v=1 << 16):
        """For each row w of W (int, (m, 56)): max over all minimal vectors u of |<w, u>|.
        The rows of W are processed in blocks of `block_w`, in `threads` Python threads
        (numpy releases the GIL inside matrix products; set the BLAS to one thread then).
        W の各行 w について、すべての最小ベクトル u にわたる |<w, u>| の最大。
        W の行を `block_w` 行ずつ、`threads` 本の Python のスレッドで処理する（numpy は行列積の中で GIL を手放す。
        そのときは BLAS を 1 スレッドにする）。"""
        GW = exact.i64_matmul(np.asarray(W, dtype=np.int64), self.G)          # (m, 56): rows w G
        # (w, u) = (w G) . u; every partial sum is bounded by sum_k |(wG)_k| |u_k|.
        # (w, u) = (w G)·u。部分和はすべて Σ_k |(wG)_k||u_k| で抑えられる。
        bound = int(np.abs(GW).sum(axis=1).max()) * self.vmax
        assert bound < 2 ** 24, f"float32 product would not be exact (bound {bound})"
        A = GW.astype(np.float32)
        V = self.V

        def run(Ab):
            buf = np.empty((len(Ab), block_v), dtype=np.float32)
            hi = np.full(len(Ab), -np.inf, dtype=np.float32)
            lo = np.full(len(Ab), np.inf, dtype=np.float32)
            for i in range(0, len(V), block_v):
                # V is kept as int16 and converted per block (exact; saves 1.75 GB).
                # V は int16 で持ち、ブロックごとに変換する（厳密。1.75 GB の節約）。
                Vb = V[i:i + block_v].astype(np.float32)
                out = buf[:, :len(Vb)]
                np.matmul(Ab, Vb.T, out=out)
                np.maximum(hi, out.max(axis=1), out=hi)
                np.minimum(lo, out.min(axis=1), out=lo)
            return np.maximum(hi, -lo)

        blocks = [A[i:i + block_w] for i in range(0, len(A), block_w)]
        if threads > 1:
            with ThreadPoolExecutor(threads) as ex:
                res = list(ex.map(run, blocks))
        else:
            res = [run(b) for b in blocks]
        r = np.concatenate(res)
        ri = np.rint(r).astype(np.int64)
        assert (ri.astype(np.float32) == r).all()
        return ri
