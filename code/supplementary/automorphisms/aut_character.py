"""S8. Conjugacy classes of Aut(R), its character on V = R tensor Q, and the Molien series
(spherical designs from the group alone) (paper §7.8; Table D2 [C26]). Not used in the proof of Theorem A.
S8. Aut(R) の共役類、V = R⊗Q 上の指標、Molien 級数（群だけから得られる球面デザイン）（論文 §7.8。表 D2 の [C26]）。
定理 A の証明には使わない。

A = Aut(R) is built modulo 3 from its generators (as in S7) and moved to a permutation group
(degree 12,992). Class representatives are written as words in the generators and multiplied out as
integer matrices, which gives exact traces. Then:
A を生成元から 3 を法として作り（S7 と同じ）、置換群（次数 12,992）に移す。共役類の代表を生成元の語として取り出し、整数行列として掛けて
正確なトレースを得る。そのうえで:
  (1) 206 conjugacy classes; <chi_V, chi_V> = 2 (consistent with End_A(V) = Q(sqrt -29));
      共役類は 206 個。<chi_V, chi_V> = 2（End_A(V) = Q(√-29) と整合）。
  (2) the Molien coefficients a_k = dim Sym^k(V)^A = (1/|A|) sum_g h_k(g) for k <= 12, where h_k is the
      complete homogeneous symmetric function of the eigenvalues, obtained from the power sums
      tr(M^j) by Newton's identity k h_k = sum_{j=1}^k p_j h_{k-j} (exact rational arithmetic);
      k ≤ 12 の Molien 係数。h_k は固有値の完全斉次対称式で、冪和 tr(M^j) から Newton の恒等式で求める（有理数の厳密な計算）。
  (3) the invariant harmonic polynomials have dimension a_k - a_{k-2}; none of degree 1, 2, 3 and 48 of
      degree 4, so the group alone makes every shell of R a spherical 3-design (not 4).
      不変な調和多項式の次元は a_k - a_{k-2}。次数 1, 2, 3 には無く、次数 4 に 48 次元ある。群だけからは R の各殻は球面 3-デザイン（4 ではない）。
The permutation group is written to out/aut_perm.g (GAP format) for further use.
置換群は out/aut_perm.g（GAP の形式）に書き出す。

Requires SageMath (GAP). Run time: about 40 minutes. Memory: about 3 GB.
SageMath（GAP）が要る。実行時間: 約 40 分。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/automorphisms/aut_character.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from common import Setup  # noqa: E402
from r112.report import Report  # noqa: E402

ORDER = 1169280


def main():
    from sage.all import GF, QQ, ZZ, identity_matrix, libgap, matrix
    ap = argparse.ArgumentParser(description="S8: classes, character and Molien series of Aut(R)")
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s8_aut_character", a.out)
    st = Setup(a.data, a.certificates)
    cent = st.centralizer()
    X, _ = st.outer_automorphism()
    Fp = GF(3)
    # A small generating set of C_A(S), with the corresponding integer matrices.
    # C_A(S) の小さな生成系と、対応する整数行列。
    Cg = libgap.Group([libgap(M.change_ring(Fp)) for M in cent])
    ckeys = {tuple(M.change_ring(Fp).list()): M for M in cent}
    cint = [ckeys[tuple(matrix(Fp, c.sage()).list())] for c in libgap.SmallGeneratingSet(Cg)]
    intgens = st.sR + cint + [X]
    intinv = [M.inverse().change_ring(ZZ) for M in intgens]
    gens = [libgap(M.change_ring(Fp)) for M in intgens]
    A = libgap.Group(gens)
    rep.check("|A| = 1169280", int(A.Size()) == ORDER)
    iso = libgap.IsomorphismPermGroup(A)
    small = libgap.SmallerDegreePermutationRepresentation(libgap.Image(iso))
    P = libgap.Image(small)
    rep.value("degree of the permutation representation", int(libgap.NrMovedPoints(P)))
    toP = libgap.CompositionMapping(small, iso)
    fromP = libgap.InverseGeneralMapping(toP)
    os.makedirs(rep.out_dir, exist_ok=True)
    with open(os.path.join(rep.out_dir, "aut_perm.g"), "w") as f:
        f.write("# Aut(R) as a permutation group (code/supplementary/automorphisms/aut_character.py)\n")
        f.write("AutR := Group(" + str(libgap.GeneratorsOfGroup(P)).replace("\n", "") + ");\n")
    cls = libgap.ConjugacyClasses(P)
    sizes = [int(libgap.Size(c)) for c in cls]
    rep.value("number of conjugacy classes", len(cls))
    rep.check("class sizes add up to |A|", sum(sizes) == ORDER)
    epi = libgap.EpimorphismFromFreeGroup(libgap.GroupWithGenerators([libgap.Image(toP, g) for g in gens]))

    def int_matrix(perm):
        """The integer matrix of a permutation, via a word in the generators (checked modulo 3).
        置換に対応する整数行列（生成元の語を経由。3 を法として確かめる）。"""
        word = libgap.LetterRepAssocWord(libgap.PreImagesRepresentative(epi, perm)).sage()
        M = identity_matrix(ZZ, 112)
        for e in word:
            M = M * (intgens[e - 1] if e > 0 else intinv[-e - 1])
        assert M.change_ring(Fp) == matrix(Fp, libgap.Image(fromP, perm).sage())
        return M

    power_sums = []
    for c in cls:
        M = int_matrix(libgap.Representative(c))
        Mj, row = identity_matrix(ZZ, 112), []
        for _ in range(a.kmax):
            Mj = Mj * M
            row.append(int(Mj.trace()))
        power_sums.append(row)
    chi = [row[0] for row in power_sums]
    norm = sum(QQ(sz) * c * c for sz, c in zip(sizes, chi)) / ORDER
    rep.value("(1) <chi_V, chi_V>", str(norm))
    rep.check("(1) <chi_V, chi_V> = 2", norm == 2)
    coeffs = []
    for k in range(a.kmax + 1):
        tot = QQ(0)
        for sz, row in zip(sizes, power_sums):
            h = [QQ(1)]
            for kk in range(1, k + 1):
                h.append(sum(row[j - 1] * h[kk - j] for j in range(1, kk + 1)) / kk)
            tot += sz * h[k]
        coeffs.append(tot / ORDER)
    rep.check("(2) the averages are integers", all(c in ZZ for c in coeffs))
    coeffs = [int(c) for c in coeffs]
    harm = [coeffs[k] - (coeffs[k - 2] if k >= 2 else 0) for k in range(a.kmax + 1)]
    rep.value("(2) Molien coefficients a_k, k = 0..kmax", coeffs)
    rep.value("(3) dimensions of invariant harmonic polynomials, k = 0..kmax", harm)
    rep.check("(3) none in degrees 1, 2, 3 and 48 in degree 4: spherical 3-designs (not 4) from the group",
              harm[1:4] == [0, 0, 0] and harm[4] == 48)
    rep.finish()


if __name__ == "__main__":
    main()
