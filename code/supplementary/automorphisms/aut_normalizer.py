"""S5. The normalizer of S = SL_2(29) in Aut(R) (paper §3.4, §7.8; Table D2 [C24]). Not used in the
proof of Theorem A.
S5. Aut(R) における S = SL_2(29) の正規化群（論文 §3.4、§7.8。表 D2 の [C24]）。定理 A の証明には使わない。

(1) Centralizer: the automorphisms of R commuting with S are found among the block matrices
    [[a, b], [c, d]] (a, b, c, d in M' tensor Q) as in common.Setup.isometries_from. Expected: 48.
    中心化群: S と可換な R の自己同型を、ブロック行列 [[a, b], [c, d]] の中から common.Setup.isometries_from で求める。期待値 48。
(2) Outer automorphism: for the outer automorphism alpha of S (GAP, on the reduction modulo 3, which
    is faithful on a finite group of integer matrices), the twisted intertwiners T with
    x g T = x T alpha(g) on L give a 32-dimensional space in the same way, and the isometries of R
    in it are searched. If found, |N_{Aut R}(S)| = 2 |S C_A(S)| = 1,169,280.
    外部自己同型: S の外部自己同型 alpha（GAP。整数行列の有限群では 3 を法とした還元は忠実）について、L 上の x g T = x T alpha(g) を満たす
    T から同じように 32 次元の空間を作り、その中の R の等長変換を探す。見つかれば |N_{Aut R}(S)| = 2|S C_A(S)| = 1,169,280。
That S is normal in Aut(R), so that this normalizer is all of Aut(R), is proved in Appendix G.
S が Aut(R) の正規部分群であり、したがってこの正規化群が Aut(R) 全体であることは付録 G で証明する。

The automorphism found in (2) is compared with certificates/outer_automorphism_R.txt.
(2) で見つけた自己同型は certificates/outer_automorphism_R.txt と比べる。

Requires SageMath (PARI, GAP). Run time: about 15 minutes. Memory: about 3 GB.
SageMath（PARI, GAP）が要る。実行時間: 約 15 分。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/automorphisms/aut_normalizer.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from common import Setup, intertwiners  # noqa: E402
from r112 import certs  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import GF, QQ, ZZ, identity_matrix, lcm, libgap, matrix
    ap = argparse.ArgumentParser(description="S5: normalizer of SL_2(29) in Aut(R)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s5_aut_normalizer", a.out)
    st = Setup(a.data, a.certificates)
    n = 56

    half, ncand, k = st.isometries_from(st.Mb)
    rep.value("(1) block matrices: lattice rank, elements with Q <= 112 (up to sign)", [k, ncand])
    rep.value("(1) |C_Aut(R)(S)|", 2 * len(half))
    rep.check("(1) the centralizer has order 48", 2 * len(half) == 48)

    F3 = GF(3)
    GB = libgap.Group([libgap(st.g1.change_ring(F3)), libgap(st.g2.change_ring(F3))])
    aut = libgap.AutomorphismGroup(GB)
    inn = libgap.InnerAutomorphismsAutomorphismGroup(aut)
    tr = libgap.RightTransversal(aut, inn)
    reps = [tr[i] for i in range(int(libgap.Size(tr)))]
    rep.value("|Out(SL_2(29))| (GAP)", len(reps))
    hom = libgap.EpimorphismFromFreeGroup(GB)
    gi = {1: st.g1, 2: st.g2}
    ginv = {1: st.g1.inverse(), 2: st.g2.inverse()}

    def lift(img):
        """An element of the mod-3 group, as an integer matrix: write it as a word in g1, g2 and multiply.
        3 を法とした群の元を、g1, g2 の語で書いて掛け、整数行列にする。"""
        word = libgap.LetterRepAssocWord(libgap.PreImagesRepresentative(hom, img)).sage()
        A = identity_matrix(ZZ, n)
        for e in word:
            A = A * (gi[e] if e > 0 else ginv[-e])
        return A.change_ring(ZZ)

    found = []
    for j, alpha in enumerate(reps):
        a1 = lift(libgap.Image(alpha, libgap(st.g1.change_ring(F3))))
        a2 = lift(libgap.Image(alpha, libgap(st.g2.change_ring(F3))))
        if a1 == st.g1 and a2 == st.g2:
            continue                                              # the trivial class / 自明な類
        Tq = intertwiners([st.g1, st.g2], [a1, a2], n)            # x g T = x T alpha(g)
        rep.value(f"(2) outer class {j}: dimension of the twisted intertwiners on L", len(Tq))
        if not Tq:
            continue
        Tz = matrix(QQ, [T.list() for T in Tq])
        Tz = matrix(ZZ, [r * lcm([x.denominator() for x in r]) for r in Tz.rows()]).saturation()
        isos, ncand, k = st.isometries_from([matrix(ZZ, n, n, list(r)) for r in Tz.rows()])
        rep.value(f"(2) outer class {j}: isometries of R (up to sign)", len(isos))
        found += isos
    rep.check("(2) the outer automorphism of SL_2(29) is induced by automorphisms of R (24 up to sign, 48 in all)", len(found) == 24)
    rep.value("|N_Aut(R)(S)| = 24360 * 48 / 2 * 2", 24360 * 48 // 2 * 2)
    # The distributed X may differ from those found here by an element of S (GAP may choose another
    # representative of the outer class). Check directly that X normalizes S and is not in S C_A(S).
    # 配布した X は、ここで見つけたものと S の元だけ違いうる（GAP が外部類の別の代表を選びうる）。X が S を正規化し、S C_A(S) に入らないことを直接確かめる。
    X = matrix(ZZ, certs.load_matrix("outer_automorphism_R.txt", 112, a.certificates).tolist())
    Xi = X.inverse()
    S3 = libgap.Group([libgap(h.change_ring(F3)) for h in st.sR])
    cent = half + [-Y for Y in half]
    SC3 = libgap.ClosureGroup(S3, libgap.Group([libgap(Y.change_ring(F3)) for Y in cent]))
    normalizes = all(bool(libgap.IN(libgap((Xi * h * X).change_ring(F3)), S3)) and all(c in ZZ for c in (Xi * h * X).list()) for h in st.sR)
    rep.check("certificates/outer_automorphism_R.txt: an isometry of R normalizing S", X * st.GR * X.transpose() == st.GR and normalizes)
    rep.check("certificates/outer_automorphism_R.txt is not in S C_A(S) (it induces the outer automorphism)",
              not bool(libgap.IN(libgap(X.change_ring(F3)), SC3)))
    rep.check("the elements found here also lie outside S C_A(S)", all(not bool(libgap.IN(libgap(Y.change_ring(F3)), SC3)) for Y in found))
    rep.finish()


if __name__ == "__main__":
    main()
