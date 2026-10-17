"""S6. The finite check excluding the "diagonal case" in the proof that SL_2(29) is normal in Aut(R)
(paper Appendix G.3; Table D2 [C25]). Not used in the proof of Theorem A.
S6. SL_2(29) が Aut(R) の正規部分群であることの証明で「対角の場合」を除く有限の検査（論文 付録 G.3。表 D2 の [C25]）。定理 A の証明には使わない。

Situation (Appendix G.3): if S = SL_2(29) sat diagonally in two components of E(A), then
V = V_1 + V_2 (S-invariant, 56-dimensional each) and s -> (s on V_1, identity on V_2) would be a
homomorphism from S into A = Aut(R). In particular the image c = (-1, +1) of s = -1 is an involution
of the centralizer C_A(S), and g -> P_- g + P_+ (P_+- = (1 +- c)/2) preserves R.
状況（付録 G.3）: S = SL_2(29) が E(A) の 2 つの成分に対角に入るなら、V = V_1 ⊕ V_2（S-不変、56 次元ずつ）で、
s -> (V_1 上で s、V_2 上で恒等) が S から A = Aut(R) への準同型になる。とくに s = -1 の像 c = (-1, +1) は C_A(S) の対合で、
g -> P_- g + P_+（P_± = (1 ± c)/2）は R を保つ。
Check: for each involution c != +-1 of C_A(S) (48 elements, recomputed here), P_- g + P_+ is not an
integral matrix in the R-basis for some generator g of S. Then the diagonal case cannot occur.
検査: C_A(S)（48 元、ここで計算し直す）の ±1 以外の各対合 c について、S のある生成元 g で P_- g + P_+ が R の基底で整数行列にならない。
すると対角の場合は起こらない。

Requires SageMath. Run time: about 1 minute. / SageMath が要る。実行時間: 約 1 分。
Run / 実行:  sage -python code/supplementary/automorphisms/aut_diagonal.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from common import Setup  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import QQ, ZZ, identity_matrix
    ap = argparse.ArgumentParser(description="S6: the diagonal case of Appendix G.3")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s6_aut_diagonal", a.out)
    st = Setup(a.data, a.certificates)
    cent = st.centralizer()
    rep.check("C_A(S) has 48 elements, all commuting with the generators of S",
              len(cent) == 48 and all(X * h == h * X for X in cent for h in st.sR))
    I2 = identity_matrix(ZZ, 112)
    invs = [X for X in cent if X * X == I2 and X != I2 and X != -I2]
    rep.value("involutions other than +-1 in C_A(S), with their traces", sorted(int(X.trace()) for X in invs))
    survivors = 0
    for c in invs:
        Pm = (I2 - c).change_ring(QQ) / 2
        Pp = (I2 + c).change_ring(QQ) / 2
        survivors += all(all(x in ZZ for x in (Pm * h + Pp).list()) for h in st.sR)
    rep.value("involutions c for which g -> P_- g + P_+ preserves R for both generators", survivors)
    rep.check("none: the diagonal case is excluded", survivors == 0)
    rep.finish()


if __name__ == "__main__":
    main()
