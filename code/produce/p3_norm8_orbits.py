"""P3. Find representatives of the G*-orbits on L_8 by the method of minimal-sum elements
(paper §6.4, Definition 6.1, Table 7). Produces the list that verify/v4_norm8_orbits.py checks.
P3. 最小和元の方法で L_8 の G*-軌道の代表を見つける（論文 §6.4、定義 6.1、表 7）。verify/v4_norm8_orbits.py が確かめる一覧を作る。

G* = G C, where C is the centralizer (certificates/centralizer.npz holds C/{+-1}); |G*| = 24,360 * 12 / 2
= 146,160. The 36,957,286,800 norm-8 vectors cannot be stored, so orbits are told apart by marking
only their minimal-sum elements:
G* = G C（C は中心化群。certificates/centralizer.npz に C/{±1}）。|G*| = 24,360 × 12 / 2 = 146,160。
369 億本のノルム 8 のベクトルは持てないので、軌道を区別するのに、最小和元にだけ印を付ける:

  * Every norm-8 vector we need is x = r + s with r, s minimal and <r, s> = -2, and by symmetry r
    can be taken among the G*-orbit representatives r_i of L_6 (150 of them).
    X_i = { r_i + s : s in L_6, <r_i, s> = -2 } (the candidates).
    必要なノルム 8 のベクトルはすべて x = r + s（r, s は最小、<r, s> = -2）と書け、対称性により r は L_6 の G*-軌道の代表 r_i
    （150 個）から取れる。X_i = { r_i + s : s ∈ L_6, <r_i, s> = -2 }（候補）。
  * The minimal-sum elements of an orbit O are the elements of O in some X_i.
    軌道 O の最小和元とは、O の元で、ある X_i に入るもの。
  * The candidates x in X_0, X_1, ... are visited in order; an unmarked x starts a new orbit, and
    all minimal-sum elements of its orbit are computed and marked at once: y = +-x u g^{-1} lies in
    X_i iff <y, r_i> = 4 iff <x u, r_i g> = +-4, so with the precomputed list W = { r_i g } a single
    matrix product finds them all.
    候補 x ∈ X_0, X_1, … を順に見る。印の無い x は新しい軌道を始め、その軌道の最小和元をすべて一度に求めて印を付ける:
    y = ±x u g^{-1} が X_i に入る ⟺ <y, r_i> = 4 ⟺ <x u, r_i g> = ±4。前もって W = { r_i g } を並べておけば、行列積 1 回で全部見つかる。
  * Speed-up: the pairs with <x u, r_i g> = +-4 correspond to decompositions x = w + w' (w, w'
    minimal, <w, w'> = -2), and when x = r_i + s0 every such w lies in {r_i, s0} or among the
    minimal vectors with inner product 3 or 2 with r_i (precomputed per i). This replaces the scan
    of W by a hash look-up (about 30 times faster). The first 25 orbits are cross-checked against
    the plain scan.
    高速化: <x u, r_i g> = ±4 となる組は分解 x = w + w'（w, w' は最小、<w, w'> = -2）に対応し、x = r_i + s0 のとき、そのような w は
    {r_i, s0} か、r_i との内積が 3 または 2 の最小ベクトル（i ごとに前もって作る）の中にある。これで W の走査がハッシュの表引きに
    置き換わる（約 30 倍速い）。最初の 25 軌道は素朴な走査と突き合わせる。
  * The stabilizer order of x in G* is the number of times x returns to itself as a minimal-sum
    element of X_i (i the index that produced x); the orbit size is |G*| / that.
    x の G* における固定群の位数は、x が X_i（i は x を作った番号）の最小和元として自分自身に戻ってくる回数。軌道の大きさ = |G*| / それ。
  * The search stops when the orbit sizes add up to |L_8| = 36,957,286,800 and three more X_i have
    been confirmed to contain no unmarked candidate.
    軌道の大きさの和が |L_8| = 36,957,286,800 に達し、さらに 3 つの X_i に印の無い候補が無いことを確かめたら止める。

Known vectors are stored as 64-bit hashes; a collision could only make the search skip an orbit,
which would show up as a missing size; the result is checked independently by V4.
既知のベクトルは 64 ビットのハッシュで持つ。衝突は軌道を飛ばす方向にしか働かず、大きさの不足として現れる。結果は V4 が独立に確かめる。

Products in float32 are exact: all entries are small integers, and the bound on the partial sums is
asserted (r112.exact / explicit asserts below).
float32 の積は厳密: 成分はすべて小さな整数で、部分和の上界を確かめる（r112.exact と下の assert）。

Checkpoint: out/p3/state.npz every 10 minutes and after each i; rerun the same command to resume.
途中経過: 10 分ごとと各 i のあとに out/p3/state.npz に保存する。同じコマンドで再開する。

Output: out/p3_norm8_reps.npz (reps int16 (k, 56), sizes int64 (k,)). It is compared with
certificates/norm8_reps.npz by the set of canonical forms of the classes x u_k (the order and the
choice of representatives may differ).
出力: out/p3_norm8_reps.npz。certificates/norm8_reps.npz とは、類 x u_k の標準形の集合で比べる（並びと代表の選び方は違いうる）。

Run time: about 1.5 hours. Memory: about 2.5 GB.
実行時間: 約 1.5 時間。メモリ: 約 2.5 GB。

Run / 実行:  python3 code/produce/p3_norm8_orbits.py [--trial N] [--data DIR] [--certificates DIR] [--out DIR]
             (--trial N stops after N orbits and reports the speed / N 軌道で止めて速さを報告する)
"""
import argparse
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import certs, data, exact, group, mod2  # noqa: E402
from r112.report import Report  # noqa: E402

N8 = 36957286800


def gstar_roots(R6, U, can):
    """Representatives of the G*-orbits on L_6: merge the 656 G-orbits r_j whose classes are
    related by an element of C (union-find on canonical forms of r_j u).
    L_6 の G*-軌道の代表: C の元で移りあう G-軌道 r_j をまとめる（r_j u の標準形による union-find）。"""
    key = {int(c): j for j, c in enumerate(can.canon(mod2.pack(R6)))}
    parent = list(range(len(R6)))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for j, r in enumerate(R6):
        for u in U:
            b = key[int(can.canon(mod2.pack(r @ u))[0])]
            ra, rb = find(j), find(b)
            parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({j for j in range(len(R6)) if parent[j] == j})
    return R6[roots]


class State:
    """Search state, saved as .npz (no pickle). / 探索の状態（pickle を使わず .npz で保存）。"""

    def __init__(self, m):
        self.i = 0
        self.reps, self.sizes, self.nx = [], [], []
        self.pending = {j: [] for j in range(m)}       # hashes of X_j-members already marked / 印の付いた X_j の元のハッシュ
        self.cur = None                                 # (i, next index, flags, new orbits) inside X_i

    def save(self, path):
        pend_idx, pend_val = [], []
        for j, lst in self.pending.items():
            if lst:
                v = np.concatenate(lst)
                self.pending[j] = [v]
                pend_idx.append(np.full(len(v), j, dtype=np.int32))
                pend_val.append(v)
        arrays = dict(
            i=np.int64(self.i),
            reps=np.array(self.reps, dtype=np.int16).reshape(-1, 56),
            sizes=np.array(self.sizes, dtype=np.int64),
            nx=np.array(self.nx, dtype=np.int64),
            pend_idx=np.concatenate(pend_idx) if pend_idx else np.zeros(0, dtype=np.int32),
            pend_val=np.concatenate(pend_val) if pend_val else np.zeros(0, dtype=np.int64),
            cur=np.array(self.cur[:2] + (self.cur[3],) if self.cur else (-1, -1, -1), dtype=np.int64),
            cur_flags=self.cur[2] if self.cur else np.zeros(0, dtype=bool),
        )
        certs.save_npz(path, **arrays)

    @classmethod
    def load(cls, path, m):
        st = cls(m)
        with np.load(path, allow_pickle=False) as z:
            st.i = int(z["i"])
            st.reps = list(z["reps"])
            st.sizes = [int(v) for v in z["sizes"]]
            st.nx = [int(v) for v in z["nx"]]
            idx, val = z["pend_idx"], z["pend_val"]
            for j in np.unique(idx):
                st.pending[int(j)] = [val[idx == j]]
            for j in range(st.i):
                st.pending[j] = None
            c = z["cur"]
            st.cur = (int(c[0]), int(c[1]), z["cur_flags"].copy(), int(c[2])) if c[0] >= 0 else None
        return st


def norm8_orbits(G, S, R6, Un, pdir, rep, n8=N8, trial=None):
    """Representatives and sizes of the G*-orbits on the norm-8 vectors of the lattice with Gram matrix G
    (G* generated by S and the centralizer elements Un), by minimal-sum elements as described above.
    Checkpoints in `pdir`. Usable for any lattice with an action of SL_2(29) (also supplementary/nebe).
    Gram 行列 G の格子のノルム 8 のベクトルの G*-軌道（G* は S と中心化群の元 Un が生成）の代表と大きさを、上の最小和元の方法で求める。
    途中経過は `pdir` に保存する。SL_2(29) が作用する任意の格子に使える（supplementary/nebe にも）。"""
    n = len(G)
    Ginv = exact.inverse_unimodular(G)
    S = np.asarray(S, dtype=np.int64)
    R6 = np.asarray(R6, dtype=np.int64)
    Un = np.asarray(Un, dtype=np.int64)
    ng = len(S)
    nu = len(Un)
    GSTAR = 2 * ng * nu
    can = mod2.Canonical(S)
    R = gstar_roots(R6, Un, can)
    m = len(R)
    rep.value("|G*|", GSTAR)
    rep.value("G*-orbits of minimal vectors (from 656 G-orbits)", m)

    # W = { r_i g }, with hashes sorted for look-up. / W = { r_i g } とその整列したハッシュ。
    S32 = S.astype(np.float32)
    smax = int(np.abs(S).max())
    assert int(np.abs(R6).sum(axis=1).max()) * smax < 2 ** 24
    W = np.rint(np.einsum("ij,gjk->igk", R.astype(np.float32), S32)).astype(np.int64).reshape(m * ng, n)
    assert np.abs(W).max() < 2 ** 15
    GR = (G @ R.T).astype(np.float32)                         # (56, m): <y, r_i> = y @ GR[:, i]
    rng = random.Random(0)
    hv = np.array([rng.getrandbits(62) | 1 for _ in range(n)], dtype=np.int64)
    hW = W @ hv
    W = W.astype(np.int16)
    oW = np.argsort(hW, kind="stable")
    hWs = hW[oW]
    Einv = np.stack([group.inverse_isometry(g, G, Ginv) for g in S])
    assert np.abs(Einv).max() < 2 ** 15
    Einv32 = Einv.astype(np.float32)
    einv_max = int(np.abs(Einv).max())
    del Einv
    rep.log(f"tables ready: W has {len(W)} rows")

    def candidates(i):
        """X_i = { r_i + s : s minimal, <r_i, s> = -2 } and N_3(r_i) = { s : <r_i, s> = 3 }.
        All minimal vectors are +-(656 representatives) g. Duplicates (from stabilizers) are removed.
        X_i と N_3(r_i)。全最小ベクトル = ±(656 代表)·g。（固定群による）重複は除く。"""
        out, n3 = [], []
        gr = GR[:, i]
        for s in R6:
            O = np.rint(s.astype(np.float32) @ S32).astype(np.int64)          # (ng, 56)
            ip = np.rint(O.astype(np.float32) @ gr).astype(np.int64)
            out.append(R[i] + O[ip == -2])
            out.append(R[i] - O[ip == 2])
            n3.append(O[ip == 3])
            n3.append(-O[ip == -3])
        Xc = np.concatenate(out)
        _, first = np.unique(Xc @ hv, return_index=True)
        N3 = np.concatenate(n3)
        _, f3 = np.unique(N3 @ hv, return_index=True)
        return Xc[np.sort(first)], N3[np.sort(f3)]

    os.makedirs(pdir, exist_ok=True)
    ckpt = os.path.join(pdir, "state.npz")
    st = State.load(ckpt, m) if (os.path.exists(ckpt) and trial is None) else State(m)
    if st.i or st.reps:
        rep.log(f"resuming at i = {st.i} with {len(st.reps)} orbits")
    t0 = time.time()
    nverify, checked, done_checks = 0, 0, 0
    UnA = Un
    for i in range(st.i, m):
        Xi, N3 = candidates(i)
        hX = Xi @ hv
        r = R[i]
        N2 = r - Xi                                             # N_2(r) = { s : <r, s> = 2 } = r - X_i
        N2G = (N2 @ G).astype(np.float32)
        N3G = (N3 @ G).astype(np.float32)
        order = np.argsort(hX)
        hs = hX[order]
        assert (np.diff(hs) != 0).all(), "hash collision inside X_i"
        flags = np.zeros(len(Xi), dtype=bool)
        if st.pending[i]:
            flags |= np.isin(hX, np.concatenate(st.pending[i]))
        new_here, start = 0, 0
        if sum(st.sizes) == n8 and st.cur is None:
            # All vectors are accounted for: confirm that X_i has no unmarked candidate.
            # 全ベクトルが数え終わっている: X_i に印の無い候補が無いことを確かめる。
            assert flags.all(), "all vectors are accounted for, but an unmarked candidate remains"
            st.i = i + 1
            st.nx.append(len(Xi))
            st.pending[i] = None
            done_checks += 1
            if done_checks >= 3:
                break
            continue
        if st.cur is not None and st.cur[0] == i:
            start, flags, new_here = st.cur[1], st.cur[2].copy(), st.cur[3]
        tck = time.time()
        for idx in range(start, len(Xi)):
            if time.time() - tck > 600:
                st.cur = (i, idx, flags.copy(), new_here)
                st.save(ckpt)
                tck = time.time()
                rep.log(f"i = {i}: {idx} / {len(Xi)} candidates, {len(st.reps)} orbits, {100 * sum(st.sizes) / n8:.2f} % of L_8")
            if flags[idx]:
                continue
            x = Xi[idx]
            XU = np.stack([x @ u for u in Un])                  # (nu, 56)
            XU32 = XU.astype(np.float32)
            # Minimal vectors w occurring in decompositions x = w + w' (<w, w'> = -2): with x = r + u0,
            # <x, w> = 4 iff (<r, w>, <u0, w>) is (6,-2), (-2,6), (3,1), (1,3) or (2,2).
            # 分解 x = w + w' に現れる最小ベクトル w。x = r + u0 のとき <x, w> = 4 ⟺ (<r,w>, <u0,w>) が上のどれか。
            u0 = x - r
            u0f = u0.astype(np.float32)
            Aw = N3[np.rint(N3G @ u0f) == 1]
            Bw = N2[np.rint(N2G @ u0f) == 2]
            Dc = np.concatenate([r[None, :], u0[None, :], Aw, x[None, :] - Aw, Bw])
            # Hits: r_{i'} g = +-(w u); look up all (i', g) by hash. / ヒット: r_{i'} g = ±(w u) をハッシュで引く。
            Z = np.einsum("dj,ujk->udk", Dc, UnA).reshape(-1, n)
            ucol = np.repeat(np.arange(nu), len(Dc))
            hZ = Z @ hv
            rows_l, cols_l, sgn_l = [], [], []
            for sg, hq in ((1, hZ), (-1, -hZ)):
                lo = np.searchsorted(hWs, hq, side="left")
                hi = np.searchsorted(hWs, hq, side="right")
                cnt = hi - lo
                if cnt.any():
                    which = np.repeat(np.arange(len(hq)), cnt)
                    offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    rows_l.append(oW[lo[which] + offs])
                    cols_l.append(ucol[which])
                    sgn_l.append(np.full(len(which), sg, dtype=np.int64))
            rows, cols, sgn = np.concatenate(rows_l), np.concatenate(cols_l), np.concatenate(sgn_l)
            if nverify < 25:
                # Cross-check against the plain scan of W. / W の素朴な走査と突き合わせる。
                P = W.astype(np.float32) @ (G @ XU.T).astype(np.float32)
                r2, c2 = np.nonzero(np.abs(P) == 4.0)
                s2 = np.sign(P[r2, c2]).astype(np.int64)
                assert sorted(zip(rows.tolist(), cols.tolist(), sgn.tolist())) == sorted(zip(r2.tolist(), c2.tolist(), s2.tolist())), \
                    "decomposition method disagrees with the plain scan"
                nverify += 1
            keep = rows >= i * ng                                # members of X_{i'} with i' >= i / i' ≥ i の X_{i'} の元
            rows, cols, sgn = rows[keep], cols[keep], sgn[keep]
            ii, gi = np.divmod(rows, ng)
            assert int(np.abs(XU).sum(axis=1).max()) * einv_max < 2 ** 24
            Y = np.rint(np.matmul(XU32[cols][:, None, :], Einv32[gi])[:, 0, :]).astype(np.int64) * sgn[:, None]   # y = +-x u g^{-1}
            if checked < 3000:
                # Spot check: y has norm 8 and y - r_{i'} has norm 6. / 抜き取り検査。
                Dd = Y - R[ii]
                assert (exact.norms(Y, G) == 8).all() and (exact.norms(Dd, G) == 6).all(), "bad minimal-sum element"
                checked += len(Y)
            hY = Y @ hv
            here = ii == i
            pos = np.searchsorted(hs, hY[here])
            assert (hs[pos] == hY[here]).all()
            flags[order[pos]] = True
            stab = int((Y[here] == x).all(axis=1).sum())
            for j in np.unique(ii[~here]).tolist():
                pj = st.pending[j]
                pj.append(hY[ii == j])
                if len(pj) > 256:                                # avoid many small arrays / 小さな配列を溜めない
                    st.pending[j] = [np.concatenate(pj)]
            assert stab >= 1 and GSTAR % stab == 0, stab
            st.reps.append(x.astype(np.int16))
            st.sizes.append(GSTAR // stab)
            new_here += 1
            if trial is not None and len(st.reps) >= trial:
                dt = time.time() - t0
                rep.value("trial", f"{len(st.reps)} orbits in {dt:.1f} s = {1000 * dt / len(st.reps):.1f} ms per orbit")
                return None, None
        st.i = i + 1
        st.nx.append(len(Xi))
        st.cur = None
        st.pending[i] = None
        rep.log(f"i = {i}: |X_i| = {len(Xi)}, new orbits {new_here}, total {len(st.reps)}, {100 * sum(st.sizes) / n8:.3f} % of L_8")
        st.save(ckpt)
    total = sum(st.sizes)
    rep.value("G*-orbits found", len(st.reps))
    rep.value("orbit sizes {size: count}", {int(k): int(v) for k, v in zip(*np.unique(st.sizes, return_counts=True))})
    rep.check("the orbit sizes add up to |L_8|", total == n8)
    return np.stack(st.reps).astype(np.int16), np.array(st.sizes, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description="P3: orbits of norm-8 vectors by minimal-sum elements")
    ap.add_argument("--trial", type=int, default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("p3_norm8_orbits" + (f"_trial{a.trial}" if a.trial else ""), a.out)
    G = data.gram(a.data)
    S = group.sl2_29_mod_sign(data.generators(a.data)).astype(np.int64)
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    Un = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)
    can = mod2.Canonical(S)
    reps, sizes = norm8_orbits(G, S, R6, Un, os.path.join(rep.out_dir, "p3"), rep, trial=a.trial)
    if reps is None:
        rep.finish()
        return
    certs.save_npz(os.path.join(rep.out_dir, "p3_norm8_reps.npz"), reps=reps, sizes=sizes)

    # Compare with the certificate: the same set of G-orbits (canonical forms of x u_k).
    # 配布した代表と比べる: 同じ G-軌道の集合（x u_k の標準形）。
    def forms(X):
        out = set()
        for u in Un:
            XU = exact.i64_matmul(X.astype(np.int64), u)
            for b in range(0, len(XU), 1 << 15):
                out |= set(can.canon(mod2.pack(XU[b:b + (1 << 15)])).tolist())
        return out
    mine = forms(reps)
    old = forms(certs.load("norm8_reps.npz", "reps", a.certificates))
    rep.check("same 1517130 G-orbits as certificates/norm8_reps.npz (equal sets of canonical forms)",
              mine == old and len(mine) == 1517130)
    rep.finish()


if __name__ == "__main__":
    main()
