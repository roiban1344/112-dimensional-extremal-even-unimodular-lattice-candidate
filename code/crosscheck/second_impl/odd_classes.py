r"""元の計算の「(6,6)・(6,10)・(10,6) 型の衝突なし」を保証する代表元（公開版では certificates/odd_bounding.npz）を、別のコードで検算する。

データ: 最小ベクトルの軌道代表 $r$（656 個）と向き（$f$, $f^{-1}$）ごとに、ベクトル $w_0$。主張は
  (i) $w_0\equiv rM \pmod{2L}$（$M=X$ または $X^{-1}$ mod 2）、(ii) $N(w_0)\in\{14,18\}$、(iii) すべての最小ベクトル $u$ について $|(w_0,u)|\le6$。
これが言えれば類 $rM+2L$ にノルム $\le10$ の元は無い（この論証もここで独立に導いた）: 類の別の元 $w$（$N(w)\le10$）があれば $y_\pm=(w_0\pm w)/2\in L\setminus\{0\}$ で
$N(y_\pm)=\tfrac14(N(w_0)+N(w)\pm2(w_0,w))\ge6$。$N(w_0)=14$: $N(w)=6$ は両立せず、$N(w)=10$ なら $(w_0,w)=0$ で $y_\pm$ は最小ベクトル、$(w_0,y_\pm)=7$。
$N(w_0)=18$: $N(w)=6$ なら $(w_0,w)=0$, $(w_0,y_\pm)=9$。$N(w)=10$ なら $(w_0,w)=\pm2$ で一方の $y$ が最小ベクトル、$(w_0,y)=8$。どれも (iii) に反する。
（ノルム 8 の $w$ は、類が「奇」（ノルム ≡ 2 mod 4）なので存在しない。）

最小ベクトル全部（15,590,400 本）は、656 個の代表と SL$_2$(29) の生成元からこのプログラムが自分で展開する。整数演算（float64 の行列積は絶対値 2^53 未満で厳密、その範囲であることも確かめる）。
実行（リポジトリの根から）: python3 code/crosscheck/second_impl/odd_classes.py

公開用の注記: 公開リポジトリ向けに変えたのは入力（data/ の 4 ファイルと
certificates/ の検証用データを読む。中心化群の元は Sage で求め直さず certificates/centralizer.npz から読む）だけで、検査そのものは変えていない。
Release note: for the public repository only the inputs were changed (the four files of data/ and the
certificate files of certificates/ are read; the centralizer elements are read from
certificates/centralizer.npz instead of being recomputed with Sage). The checks are unchanged.
Shares no code with code/r112 or code/verify. Needs only numpy.
"""
import sys, time
from pathlib import Path
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
n = 56; t0 = time.time()
Gn = np.array(g_, dtype=np.int64).reshape(n, n); H = [np.array(h, dtype=np.int64).reshape(n, n) for h in h_]
X = _ints(DATA / "tau.txt").reshape(n, n)
assert ((X @ X - np.eye(n, dtype=np.int64)) % 2 == 0).all()          # tau^2 = -5 = 1 (mod 2): tau^{-1} = tau mod 2
Ms = [X, X % 2]
R = _npz("min_reps.npz", "reps").astype(np.int64)
W0 = _npz("odd_bounding.npz", "w0").astype(np.int64); rep = np.arange(len(W0)); which = np.zeros(len(W0), dtype=np.int64)
pairs = sorted(zip(rep.tolist(), which.tolist()))
involutive = ((Ms[0] @ Ms[0] - np.eye(n, dtype=np.int64)) % 2 == 0).all()      # X^2 ≡ 1 (mod 2) なら f = f^{-1} で、片方の向きで足りる
need = {(i, j) for i in range(len(R)) for j in ((0,) if involutive else (0, 1))}
print(f"X^2 ≡ 1 (mod 2): {involutive}。代表元 {len(W0)} 件、必要な (代表, 向き) {len(need)} 組を尽くす: {set(pairs) == need}", flush=True)
assert set(pairs) == need
# (i), (ii)
cong = all(((W0[k] - R[rep[k]] @ Ms[which[k]]) % 2 == 0).all() for k in range(len(W0)))
norms = ((W0 @ Gn) * W0).sum(axis=1)
print(f"(i) w0 ≡ r·M (mod 2L): {cong}; (ii) ノルムの内訳 { {int(v): int((norms == v).sum()) for v in np.unique(norms)} }", flush=True)
assert cong and set(np.unique(norms).tolist()) <= {14, 18}
# 全最小ベクトルを展開（± の片方）
def key(g):
    i = np.flatnonzero(g.ravel())[0]
    return (g if g.ravel()[i] > 0 else -g).astype(np.int16).tobytes()
E0 = np.eye(n, dtype=np.int64); seen = {key(E0): E0}; frontier = [E0]
while frontier:
    new = []
    for g in frontier:
        for h in H:
            p = g @ h; k = key(p)
            if k not in seen:
                seen[k] = p; new.append(p)
    frontier = new
S = np.stack(list(seen.values())); assert len(S) == 12180
keys = set()
for r in R:
    Y = r @ S
    sgn = np.sign(Y[np.arange(len(Y)), (Y != 0).argmax(axis=1)])[:, None]
    keys |= {row.tobytes() for row in (Y * sgn).astype(np.int16)}
V = np.frombuffer(b"".join(keys), dtype=np.int16).reshape(-1, n); del keys
assert 2 * len(V) == 15590400
print(f"最小ベクトル {2 * len(V)} 本を展開  [{round(time.time() - t0)} s]", flush=True)
# (iii)
GW = (Gn @ W0.T).astype(np.float64)                     # (56, 1312)
assert np.abs(GW).max() * np.abs(V).max() * n < 2 ** 52, "float64 で厳密と言えない"
mx = np.zeros(len(W0), dtype=np.int64); hist = {}
for i in range(0, len(V), 40000):
    P = np.abs(np.rint(V[i:i + 40000].astype(np.float64) @ GW)).astype(np.int64)
    mx = np.maximum(mx, P.max(axis=0))
print(f"(iii) 各 w0 と全最小ベクトルとの内積の絶対値の最大: 内訳 { {int(v): int((mx == v).sum()) for v in np.unique(mx)} }  [{round(time.time() - t0)} s]", flush=True)
print(f"RESULT: 代表元は {'すべて有効' if (mx <= 6).all() else '無効なものがある'}（{len(W0)} 件、内積の最大 {int(mx.max())} ≤ 6 が必要）", flush=True)
