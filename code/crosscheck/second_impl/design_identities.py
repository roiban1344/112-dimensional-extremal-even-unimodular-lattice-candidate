r"""元の計算（公開版では produce/p3 と verify/v4）の「ノルム 8 の軌道代表の一覧は漏れなく重複なし」を、別の原理で点検する。

元の計算の完全性の根拠は「軌道の大きさの合計 = 36,957,286,800（テータ級数の係数）」。これは、同じ軌道を 2 回数えて別の軌道を 1 つ落としても、
大きさが同じなら成り立ってしまう。そこで球面デザインの恒等式を使う: 56 次元の極値格子の各殻は 7-デザイン [既知: Venkov] なので、任意の $a$ について
$$\sum_{N(x)=8}(x,a)^{2}=N_8\frac{8\,(a,a)}{56},\quad \sum(x,a)^4=N_8\frac{3\cdot8^2(a,a)^2}{56\cdot58},\quad \sum(x,a)^6=N_8\frac{15\cdot8^3(a,a)^3}{56\cdot58\cdot60}.$$
左辺を「代表 × 群の元」で計算する: $\sum_{y\in\mathrm{orb}(x)}(y,a)^k=\frac{|\mathrm{orb}|}{|G^*|}\sum_{g\in G^*}(xg,a)^k$。
重複や欠落があれば、整数の等式が（偶然を除いて）崩れる。群の元は生成元から自分で列挙し、元の計算のコードは読み込まない
（共有するのはデータ: Gram 行列、生成元、可換子の基底、代表と軌道の大きさ）。

実行（リポジトリの根から）: python3 code/crosscheck/second_impl/design_identities.py [a の本数]

公開用の注記: 公開リポジトリ向けに変えたのは入力（data/ の 4 ファイルと
certificates/ の検証用データを読む。中心化群の元は Sage で求め直さず certificates/centralizer.npz から読む）だけで、検査そのものは変えていない。
Release note: for the public repository only the inputs were changed (the four files of data/ and the
certificate files of certificates/ are read; the centralizer elements are read from
certificates/centralizer.npz instead of being recomputed with Sage). The checks are unchanged.
Shares no code with code/r112 or code/verify. Needs only numpy.
"""
import sys, time
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
DATA, CERT = ROOT / "data", ROOT / "certificates"


def _ints(path):
    return np.array([int(t) for l in open(path) if not l.startswith("#") for t in l.split()], dtype=np.int64)


def _npz(name, key):
    with np.load(CERT / name, allow_pickle=False) as z:
        return z[key]


g_ = _ints(DATA / "L56_gram.txt")
h_ = _ints(DATA / "sl2_29_generators.txt").reshape(2, -1)
NA = int(sys.argv[1]) if len(sys.argv) > 1 else 3
n = 56; t0 = time.time()
Gn = np.array(g_, dtype=np.int64).reshape(n, n)
H = [np.array(h, dtype=np.int64).reshape(n, n) for h in h_]
units = [u.astype(np.int64) for u in _npz("centralizer.npz", "units")]
assert all((u @ Gn @ u.T == Gn).all() for u in units) and all((h @ Gn @ h.T == Gn).all() for h in H)

# SL_2(29)/±1 を生成元から幅優先で列挙
def key(g):
    i = np.flatnonzero(g.ravel())[0]
    return (g if g.ravel()[i] > 0 else -g).astype(np.int8).tobytes()
E0 = np.eye(n, dtype=np.int64); seen = {key(E0): E0}; frontier = [E0]
while frontier:
    new = []
    for g in frontier:
        for h in H:
            p = g @ h; k = key(p)
            if k not in seen:
                seen[k] = p; new.append(p)
    frontier = new
S = np.stack(list(seen.values()))
# 単数（± の片方）が群をなすこと、SL_2(29) と可換なことを確かめる
ukeys = {key(u) for u in units}
assert all(key(u @ v) in ukeys for u in units for v in units), "単数が ± を除いて群をなさない"
assert all((u @ h == h @ u).all() for u in units for h in H)
print(f"|SL_2(29)/±| = {len(S)}, 単数（±の片方）{len(units)} 個  [{round(time.time() - t0)} s]", flush=True)
assert len(S) == 12180
ord_mod_sign = len(S) * len(units)                  # |G^*/±|（S∩単数 = {±1} を仮定。下で軌道の大きさと照合）

R = _npz("norm8_reps.npz", "reps").astype(np.int64); sizes = [int(s) for s in _npz("norm8_reps.npz", "sizes")]
N8 = 36957286800                                     # 極値テータ級数の q^4 の係数（code/verify/v2_theta.py）
assert ((R @ Gn) * R).sum(axis=1).tolist() == [8] * len(R), "ノルム 8 でない代表がある"
print(f"代表 {len(R)} 個はすべてノルム 8。大きさの合計 {sum(sizes)}（テータ級数の係数 {N8}）: {sum(sizes) == N8}; 最大の軌道 {max(sizes)}（2|G^*/±| = {2 * ord_mod_sign}）", flush=True)
assert all((2 * ord_mod_sign) % s == 0 for s in sizes)

rng = np.random.default_rng(20261013)
for trial in range(NA):
    a = rng.integers(-1, 2, size=n) if trial else np.eye(n, dtype=np.int64)[0]
    Na = int(a @ Gn @ a)
    W = np.concatenate([(S @ u) @ (Gn @ a) for u in units])          # (x g, a) = x·(g G a^T), g = s u
    Wf = W.astype(np.float64).T
    tot = {2: 0, 4: 0, 6: 0}
    for i in range(0, len(R), 400):
        P = np.rint(R[i:i + 400].astype(np.float64) @ Wf).astype(np.int64)     # |値| は小さく、float64 で厳密
        P2 = P * P
        for k, Pk in ((2, P2), (4, P2 * P2), (6, P2 * P2 * P2)):
            sk = Pk.sum(axis=1)
            tot[k] += sum(sz * int(v) for sz, v in zip(sizes[i:i + 400], sk))
    exp = {2: Fraction(N8 * 8 * Na, 56), 4: Fraction(N8 * 3 * 64 * Na ** 2, 56 * 58), 6: Fraction(N8 * 15 * 512 * Na ** 3, 56 * 58 * 60)}
    for k in (2, 4, 6):
        lhs = Fraction(tot[k], ord_mod_sign)
        print(f"  a#{trial} (N(a) = {Na}), 次数 {k}: Σ = {lhs}; デザインの値 {exp[k]}; 一致 {lhs == exp[k]}  [{round(time.time() - t0)} s]", flush=True)
