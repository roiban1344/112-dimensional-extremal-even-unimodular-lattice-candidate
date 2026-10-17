"""S7. The structure of Aut(R) from explicit generators (paper §7.8, Theorem B; Table D2 [C26]).
Not used in the proof of Theorem A.
S7. 明示的な生成元からの Aut(R) の構造（論文 §7.8、定理 B。表 D2 の [C26]）。定理 A の証明には使わない。

Generators, as 112 x 112 integer matrices in the basis (3.1): the two generators of S = SL_2(29),
the 48 elements of C_A(S) (recomputed), and the automorphism X of certificates/outer_automorphism_R.txt.
The reduction modulo 3 is injective on a finite group of integer matrices (Minkowski), so the group is
studied as a matrix group over GF(3) in GAP.
生成元（基底 (3.1) に関する 112 × 112 の整数行列）: S = SL_2(29) の 2 つの生成元、C_A(S) の 48 元（計算し直す）、
certificates/outer_automorphism_R.txt の自己同型 X。整数行列の有限群では 3 を法とした還元は単射（Minkowski）なので、GF(3) 上の行列群として GAP で調べる。

Computed / 求めるもの:
  (1) |A| = |<S, C_A(S), X>| = 1,169,280;  (2) the structure of C_A(S) and of A/S;
  (3) X^2 in S C_A(S), and the elements of <C_A(S), X> outside S C_A(S);  (4) the centre of A;
  (5) determinants of the generators (A in SO(112));
  (6) the extension does not split: among the lifts a c (c in C_A(S)) of an element a = X s whose
      image in A/C_A(S) is an involution, none is an involution;
  (7) dim_Q End_A(V) = 2 and End_{S C_A(S)}(V) = Q(i, sqrt 29) (4-dimensional, commutative);
  (8) how many of the 48 elements X c (c in C_A(S)) are block-diagonal or anti-diagonal in the
      coordinates (x, v) (8 and 8: the outer automorphism can be realized preserving the frame).
  (1) |A| = |<S, C_A(S), X>| = 1,169,280。(2) C_A(S) と A/S の構造。(3) X^2 ∈ S C_A(S) と、<C_A(S), X> の S C_A(S) の外の元。(4) A の中心。
  (5) 生成元の行列式（A ⊂ SO(112)）。(6) 拡大は分裂しない: A/C_A(S) で対合になる a = X s の持ち上げ a c（c ∈ C_A(S)）に対合は無い。
  (7) dim_Q End_A(V) = 2、End_{S C_A(S)}(V) = Q(i, √29)（4 次元、可換）。(8) 48 個の X c（c ∈ C_A(S)）のうち座標 (x, v) でブロック対角・反対角になるものの数（8 と 8: 外部自己同型は枠を保つ形でも実現できる）。

Requires SageMath (GAP). Run time: about 30 minutes. Memory: about 3 GB.
SageMath（GAP）が要る。実行時間: 約 30 分。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/automorphisms/aut_structure.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from common import Setup, commutant_in_end_S  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import GF, QQ, libgap, matrix
    ap = argparse.ArgumentParser(description="S7: structure of Aut(R)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s7_aut_structure", a.out)
    st = Setup(a.data, a.certificates)
    cent = st.centralizer()
    X, _ = st.outer_automorphism()
    rep.check("all generators are isometries of R", all(M * st.GR * M.transpose() == st.GR for M in cent + st.sR + [X]))
    rep.value("(5) determinants: S, C_A(S), X", [sorted({int(M.det()) for M in st.sR}), sorted({int(M.det()) for M in cent}), int(X.det())])

    F3 = GF(3)
    gp = lambda Ms: libgap.Group([libgap(M.change_ring(F3)) for M in Ms])   # noqa: E731
    S, C, A, K = gp(st.sR), gp(cent), gp(st.sR + cent + [X]), gp(cent + [X])
    rep.value("|S|, |C_A(S)|, |<C_A(S), X>|", [int(S.Size()), int(C.Size()), int(K.Size())])
    rep.check("(1) |A| = 1169280", int(A.Size()) == 1169280)
    rep.value("(2) C_A(S): StructureDescription", str(C.StructureDescription()))
    # <C, X> is small (order 1440): decide membership in S element by element and form the quotient
    # in its regular permutation representation (the matrix-group Intersection is slow).
    # <C, X> は小さい（位数 1440）。S への所属を元ごとに判定し、正則置換表現で商を作る（行列群の Intersection は遅い）。
    Kel = list(K.AsList())
    inS = [bool(libgap.IN(k, S)) for k in Kel]
    hom = libgap.ActionHomomorphism(K, K.AsList(), libgap.OnRight)
    Kp = libgap.Image(hom)
    KSp = libgap.Group([libgap.Image(hom, k) for k, b in zip(Kel, inS) if b])
    Q = libgap.FactorGroup(Kp, KSp)
    rep.value("(2) A/S = <C, X>S/S: order and StructureDescription", [int(Q.Size()), str(Q.StructureDescription())])
    SC = libgap.ClosureGroup(S, C)
    Xg = libgap(X.change_ring(F3))
    rep.check("(3) X^2 lies in S C_A(S)", bool(libgap.IN(Xg ** 2, SC)))
    outside = [k for k in Kel if not libgap.IN(k, SC)]
    rep.value("(3) elements of <C, X> outside S C_A(S): number and element orders",
              [len(outside), sorted({int(libgap.Order(k)) for k in outside})])
    rep.value("(4) |Z(A)|", int(A.Centre().Size()))

    # (6) splitting. / 分裂。
    Cset = {tuple(M.change_ring(F3).list()) for M in cent}
    found = None
    for s in S.AsList():
        a_ = Xg * s
        if tuple(matrix(F3, (a_ * a_).sage()).list()) in Cset:
            found = matrix(F3, a_.sage())
            break
    rep.check("(6) an element a = X s with a^2 in C_A(S) exists", found is not None)
    invols = [c for c in cent if (found * c.change_ring(F3)) ** 2 == 1]
    rep.value("(6) orders of the 48 lifts a c", sorted({int(libgap.Order(libgap(found * c.change_ring(F3)))) for c in cent}))
    rep.check("(6) no lift is an involution: the extension does not split", len(invols) == 0)

    # (7) endomorphism algebras. / 自己準同型環。
    KSC, gens32 = commutant_in_end_S(st, cent)
    KA, _ = commutant_in_end_S(st, cent + [X])
    rep.value("(7) dim End_{S C_A(S)}(V), dim End_A(V)", [KSC.nrows(), KA.nrows()])
    YSC = [sum(c * g for c, g in zip(row, gens32)) for row in KSC.rows()]
    rep.check("(7) End_A(V) has dimension 2 and End_{S C_A(S)}(V) is commutative of dimension 4",
              KA.nrows() == 2 and KSC.nrows() == 4 and all(Y * W == W * Y for Y in YSC for W in YSC))
    rep.value("(7) minimal polynomials of a basis of End_{S C_A(S)}(V)", [str(Y.minimal_polynomial()) for Y in YSC])

    # (8) frames. / 枠。
    n = 56
    kinds = []
    for c in cent:
        Y = st.Bi * (X * c).change_ring(QQ) * st.B
        b_, c_ = Y[:n, n:], Y[n:, :n]
        a0, d0 = Y[:n, :n], Y[n:, n:]
        kinds.append("diag" if b_ == 0 and c_ == 0 else "anti" if a0 == 0 and d0 == 0 else "other")
    rep.value("(8) X c in ambient coordinates: block-diagonal, anti-diagonal, other",
              [kinds.count("diag"), kinds.count("anti"), kinds.count("other")])
    rep.check("(8) 8 block-diagonal and 8 anti-diagonal", kinds.count("diag") == 8 and kinds.count("anti") == 8)
    rep.finish()


if __name__ == "__main__":
    main()
