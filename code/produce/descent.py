"""The descent that finds bounding representatives (paper §6.6). Shared by p4 and p5.
下界代表元を見つける降下（論文 §6.6）。p4 と p5 が共有する。

Start from a vector w of the class (r tau, or x u tau). While some minimal vector u has
<w, u> >= 7, replace w by w - 2u (for the u with the largest |<w, u>|, sign adjusted). This stays
in the class w + 2L and lowers the norm by 4(<w, u> - 6) > 0, so it terminates. If it ends with
|<w, u>| <= 6 for all minimal u and a suitable norm, w is a bounding representative.
類の元 w（r tau か x u tau）から始める。<w, u> ≥ 7 となる最小ベクトル u がある限り、w を w - 2u に置き換える
（|<w, u>| が最大の u を、符号を合わせて使う）。類 w + 2L の中に留まり、ノルムは 4(<w, u> - 6) > 0 だけ下がるので、必ず止まる。
すべての最小ベクトル u について |<w, u>| ≤ 6 で、ノルムが適切なら、w は下界代表元。

The inner products with all minimal vectors are P[s, j] = <r_j s, w> = r_j (s G w^T): first
t = S (G w^T) (12,180 x 56), then t R^T (12,180 x 656). Both products are done in float64 after
asserting that all partial sums are integers below 2^52, so they are exact. Ties are broken by
the first position in row-major order of P, which makes the result deterministic.
すべての最小ベクトルとの内積は P[s, j] = <r_j s, w> = r_j (s G w^T)。まず t = S (G w^T)、次に t R^T。どちらの積も、部分和が
すべて 2^52 未満の整数であることを確かめてから float64 で計算するので厳密。同点は P の行優先で最初の位置を取るので、結果は決定的。

Whatever this heuristic outputs is checked independently (verify/v5, verify/v7).
この発見的な手続きの出力は、独立に確かめる（verify/v5, verify/v7）。
"""
import numpy as np


class Descent:
    def __init__(self, G, S, R6):
        self.G = np.asarray(G, dtype=np.int64)
        self.E = np.asarray(S).astype(np.float64)              # (12180, 56, 56)
        self.R = np.asarray(R6, dtype=np.int64)                # (656, 56)
        self.RT = np.ascontiguousarray(self.R.T.astype(np.float64))
        self.emax = float(np.abs(self.E).max())
        self.rmax = float(np.abs(self.R).max())

    def inner_products(self, w):
        """P[s, j] = <r_j s, w> for all s, j (float64 holding exact integers).
        すべての s, j についての P[s, j] = <r_j s, w>（厳密な整数を保持する float64）。"""
        gw = (self.G @ w).astype(np.float64)
        assert self.emax * np.abs(gw).sum() < 2.0 ** 52
        t = self.E @ gw                                         # (12180, 56)
        assert self.rmax * np.abs(t).sum(axis=1).max() < 2.0 ** 52
        return t @ self.RT                                      # (12180, 656)

    def descend(self, w, limit):
        """Run the descent from w; returns (w, max |<w, u>|, number of steps).
        w から降下を走らせ、(w, max|<w, u>|, 段数) を返す。"""
        w = np.asarray(w, dtype=np.int64).copy()
        for step in range(limit + 1):
            P = self.inner_products(w)
            k = int(np.abs(P).argmax())
            s, j = divmod(k, P.shape[1])
            m = int(round(P[s, j]))
            if abs(m) <= 6 or step == limit:
                return w, abs(m), step
            u = np.rint(self.R[j] @ self.E[s]).astype(np.int64)  # the minimal vector r_j s / 最小ベクトル r_j s
            w = w - 2 * (1 if m > 0 else -1) * u
