"""qfminim with a completeness check. Requires SageMath.
完全性を確かめた qfminim。SageMath が要る。

PARI's qfminim enumerates the vectors of norm <= B of a positive definite form, but its
documentation does not state that the enumeration is guaranteed complete (it uses floating point).
Wherever a claim depends on having found *all* such vectors (the centralizer of SL_2(29) in Aut(L)
and in Aut(R), the identification with Nebe's lattice), this function repeats the enumeration with
the interval-arithmetic enumerator of verify/enum_interval.c, which cannot miss vectors, and asserts
that both give the same set (up to sign).
PARI の qfminim は正定値形式のノルム B 以下のベクトルを列挙するが、文書には列挙が必ず完全であるとは書かれていない（浮動小数点を使う）。
「そのようなベクトルを**すべて**見つけた」ことに依存する主張（Aut(L)・Aut(R) における SL_2(29) の中心化群、Nebe の格子との同定）では、
この関数が、取りこぼしの無い verify/enum_interval.c の区間演算の列挙で同じ列挙をやり直し、両者が（符号を除いて）同じ集合を与えることを確かめる。

The form is first LLL-reduced with PARI's qflllgram; the basis change U is checked to be unimodular,
and the vectors are mapped back. (The reduction only affects the running time.)
形式はまず PARI の qflllgram で LLL 簡約する。基底変換 U が行列式 ±1 であることを確かめ、ベクトルを元の座標に戻す（簡約は実行時間にだけ影響する）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from r112 import enumerate as exact_enum  # noqa: E402
from r112.report import DEFAULT_OUT  # noqa: E402


def qfminim_checked(Q, B):
    """Vectors c (one of each +-pair) with c Q c^T <= B for a positive definite integral Sage matrix Q:
    the PARI result, after checking it against the exact enumeration. Returns a list of Sage vectors.
    正定値な整数の Sage 行列 Q について c Q c^T ≤ B となる c（±の片方）。PARI の結果を厳密な列挙と照合してから返す。"""
    from sage.all import ZZ, matrix, pari, vector
    vecs = pari(Q).qfminim(B, None, 0)[2]
    pari_list = matrix(ZZ, vecs).transpose().rows() if vecs.ncols() else []
    U = pari(Q).qflllgram().sage().transpose()           # rows = reduced basis / 行が簡約基底
    assert abs(U.det()) == 1
    Qr = U * Q * U.transpose()
    found, _ = exact_enum.short_vectors([[int(v) for v in row] for row in Qr.rows()], int(B),
                                        os.path.join(DEFAULT_OUT, "build"))
    exact_list = [vector(ZZ, c) * U for c, _ in found]
    key = lambda v: tuple(v) if next(x for x in v if x) > 0 else tuple(-v)   # noqa: E731
    assert {key(v) for v in pari_list} == {key(v) for v in exact_list}, \
        f"qfminim returned {len(pari_list)} vectors, the exact enumeration {len(exact_list)}"
    return pari_list
