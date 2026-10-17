"""Classes modulo 2L as 56-bit integers, and canonical forms under SL_2(29) (paper §6.5).
2L を法とする類を 56 ビットの整数で表し、SL_2(29) による標準形を求める（論文 §6.5）。

A class c = x + 2L is identified with the vector x mod 2 in F_2^56 and stored as the
integer sum_i (x_i mod 2) 2^i. An element s of G/{+-1} acts F_2-linearly by c -> c s
(-1 acts trivially modulo 2). If row i of s mod 2 is packed as the integer r_i(s), then
c s = XOR of r_i(s) over the bits i set in c.
類 c = x + 2L を F_2^56 のベクトル x mod 2 と同一視し、整数 Σ_i (x_i mod 2) 2^i として持つ。
G/{±1} の元 s は c -> c s で F_2-線形に作用する（-1 は 2 を法として自明に作用する）。s mod 2 の第 i 行を整数 r_i(s) に詰めると、
c s は c の立っているビット i についての r_i(s) の XOR。

To make this fast, the XOR of the rows is tabulated per byte of c:
    T[s, b, v] = XOR of r_{8b+k}(s) over the bits k of the byte value v (0 <= v < 256),
so that c s = XOR_{b=0..6} T[s, b, byte_b(c)]: 7 look-ups per element s.
高速にするため、c のバイトごとに行の XOR を表にしておく:
    T[s, b, v] = バイト値 v（0 ≤ v < 256）の立っているビット k についての r_{8b+k}(s) の XOR。
すると c s = XOR_{b=0..6} T[s, b, (c の第 b バイト)]。1 元あたり表引き 7 回。

Definition 6.2: the canonical form can(c) is the smallest integer in the orbit {c s}.
定義 6.2: 標準形 can(c) は軌道 {c s} の中で最小の整数。

Everything is exact integer arithmetic (uint64 XOR and comparisons).
すべて整数の厳密な演算（uint64 の XOR と比較）。
"""
import numpy as np

N = 56
_BITS = np.uint64(1) << np.arange(N, dtype=np.uint64)
_SHIFTS = np.uint64(8) * np.arange(7, dtype=np.uint64)


def pack(V):
    """Integer vectors (last axis of length 56) -> classes modulo 2L as uint64.
    整数ベクトル（最後の軸の長さ 56）→ 2L を法とする類（uint64）。"""
    return ((np.asarray(V) % 2).astype(np.uint64) * _BITS).sum(axis=-1, dtype=np.uint64)


def unpack(c):
    """uint64 classes -> 0/1 vectors (int64, last axis of length 56). / 類 → 0/1 ベクトル。"""
    c = np.asarray(c, dtype=np.uint64)
    return ((c[..., None] >> np.arange(N, dtype=np.uint64)) & np.uint64(1)).astype(np.int64)


class Canonical:
    """Canonical forms of classes under a finite group of matrices S (e.g. SL_2(29)/{+-1}).
    行列の有限集合 S（例: SL_2(29)/{±1}）による類の標準形。"""

    def __init__(self, S):
        S = np.asarray(S)
        self.size = len(S)
        rows = pack(S % 2)                      # (|S|, 56): row i of s, packed / s の第 i 行を詰めたもの
        # Layout (7, 256, |S|): a look-up T[b, v] is one contiguous row over all s (fast gathers).
        # 配置 (7, 256, |S|): 表引き T[b, v] はすべての s にわたる連続した 1 行（集める操作が速い）。
        T = np.zeros((7, 256, self.size), dtype=np.uint64)
        for b in range(7):
            for v in range(1, 256):
                # Build the table from the value with the lowest bit removed.
                # 最下位ビットを除いた値の表から作る。
                low = v & -v
                k = low.bit_length() - 1
                T[b, v] = T[b, v ^ low] ^ rows[:, 8 * b + k]
        self.T = T

    def images(self, c):
        """Images c s of the classes c (uint64 array of length m) under all s: shape (m, |S|).
        類 c（長さ m の uint64 配列）のすべての s による像 c s。形は (m, |S|)。"""
        c = np.atleast_1d(np.asarray(c, dtype=np.uint64))
        by = ((c[None, :] >> _SHIFTS[:, None]) & np.uint64(255)).astype(np.intp)   # (7, m)
        out = self.T[0][by[0]]                                                     # (m, |S|), a copy
        for b in range(1, 7):
            out ^= self.T[b][by[b]]
        return out

    def canon(self, c, batch=512):
        """Canonical forms (uint64 array) of the classes c. / 類 c の標準形（uint64 の配列）。"""
        c = np.atleast_1d(np.asarray(c, dtype=np.uint64))
        out = np.empty(len(c), dtype=np.uint64)
        for i in range(0, len(c), batch):
            out[i:i + batch] = self.images(c[i:i + batch]).min(axis=1)
        return out

    def canon_and_orbit_size(self, c, batch=512):
        """Canonical forms and orbit sizes |{c s : s in S}| = |S| / |{s : c s = c}| (orbit-stabilizer).
        標準形と軌道の大きさ |{c s}| = |S| / |{s : c s = c}|（軌道・固定群の定理）。"""
        c = np.atleast_1d(np.asarray(c, dtype=np.uint64))
        can = np.empty(len(c), dtype=np.uint64)
        size = np.empty(len(c), dtype=np.int64)
        for i in range(0, len(c), batch):
            img = self.images(c[i:i + batch])
            can[i:i + batch] = img.min(axis=1)
            stab = (img == c[i:i + batch, None]).sum(axis=1)
            assert (self.size % stab == 0).all()
            size[i:i + batch] = self.size // stab
        return can, size

    def self_test(self, S, rng, trials=40):
        """Table look-up agrees with the matrix product; canonical forms are invariant.
        表引きが行列の積と一致すること、標準形が群の作用で不変なことを確かめる。"""
        S = np.asarray(S)
        v = rng.integers(0, 2, size=(trials, N))
        idx = rng.integers(0, self.size, size=trials)
        naive = pack(np.einsum("ij,ijk->ik", v, S[idx].astype(np.int64)) % 2)
        assert (self.images(pack(v))[np.arange(trials), idx] == naive).all(), "table look-up disagrees with matrix product"
        assert (self.canon(naive) == self.canon(pack(v))).all(), "canonical form is not invariant"
