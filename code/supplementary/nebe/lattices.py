"""S17. Nebe's two lattices L_{56,2}(M) and L_{56,2}(M~) with SL_2(29), built from the definition
(cycloquat.py), saved for the other scripts of supplementary/nebe. Requires SageMath.
S17. Nebe の 2 つの格子 L_{56,2}(M) と L_{56,2}(M~)（SL_2(29) つき）を定義から作り（cycloquat.py）、supplementary/nebe のほかのスクリプトのために保存する。
SageMath が要る。

H has two conjugacy classes of maximal orders (checked with the mass formula). For each, the lattice
L_{56,2} and the generators z (multiplication by zeta, order 29) and w (the Weyl element) of SL_2(29) are
built; the Weyl element is computed numerically and then checked exactly (integral on the lattice,
preserving the Gram matrix, w^2 = -1, (z w)^3 = +-1). The class whose centralizer of SL_2(29) has order
24 gives L_{56,2}(M), the one with 12 gives L_{56,2}(M~) [Neb98, Table II]. Each lattice is checked to be
even unimodular.
H の極大整環の共役類は 2 つ（質量公式で確かめる）。それぞれについて格子 L_{56,2} と SL_2(29) の生成元 z（ζ 倍、位数 29）、w（Weyl 元）を作る。
Weyl 元は数値計算で求めてから厳密に確かめる（格子上で整数、Gram 行列を保つ、w^2 = -1、(zw)^3 = ±1）。SL_2(29) の中心化群の位数が 24 の類が
L_{56,2}(M)、12 の類が L_{56,2}(M~) [Neb98, Table II]。どちらも偶ユニモジュラーであることを確かめる。

Output in out/nebe/: M_gram.txt, M_generators.txt (z, w), Mtilde_gram.txt, Mtilde_generators.txt,
and *_order.txt, *_basis.txt (the maximal order in H and the lattice basis in Q(zeta)^2, for naming tau).
出力（out/nebe/）: M_gram.txt、M_generators.txt（z, w）、Mtilde_gram.txt、Mtilde_generators.txt、*_order.txt、*_basis.txt（H の中の極大整環と Q(ζ)^2 の中の格子の基底。tau の名指しに使う）。

Run time: about 15 minutes. Memory: about 2 GB.
実行時間: 約 15 分。メモリ: 約 2 GB。
Run / 実行:  sage -python code/supplementary/nebe/lattices.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))
from r112.report import Report  # noqa: E402


def write_mats(path, header, mats):
    with open(path, "w") as f:
        for line in header.splitlines():
            f.write("# " + line + "\n")
        for k, A in enumerate(mats):
            if k:
                f.write("\n")
            for r in A.rows():
                f.write(" ".join(str(x) for x in r) + "\n")


def read_mats(path, n, ring=None):
    """Matrices from a text file (integers, or rationals written as a/b). / テキストファイルの行列（整数または a/b の有理数）。"""
    from sage.all import QQ, ZZ, matrix
    ring = ring or ZZ
    flat = [ring(t) for line in open(path) if not line.startswith("#") for t in line.split()]
    return [matrix(ring, n, n, flat[k * n * n:(k + 1) * n * n]) for k in range(len(flat) // (n * n))]


def load(name, out_dir):
    """(Gram matrix, [z, w]) of 'M' or 'Mtilde' from out/nebe. / out/nebe から 'M' または 'Mtilde' の (Gram 行列, [z, w])。"""
    d = os.path.join(out_dir, "nebe")
    return read_mats(os.path.join(d, f"{name}_gram.txt"), 56)[0], read_mats(os.path.join(d, f"{name}_generators.txt"), 56)


def main():
    from sage.all import QQ, ZZ, block_matrix, identity_matrix, matrix, zero_matrix
    from cycloquat import CycloQuat, maximal_order_classes, weyl_element
    from p1_centralizer import commutant_basis, elements_of_norm
    ap = argparse.ArgumentParser(description="S17: Nebe's lattices from the definition")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s17_nebe_lattices", a.out)
    out = os.path.join(rep.out_dir, "nebe")
    os.makedirs(out, exist_ok=True)
    X = CycloQuat(29)
    n, h = 56, 28
    Mz = matrix(QQ, [(X.z ** (k + 1)).list() for k in range(h)])
    Z = zero_matrix(QQ, h, h)
    Zr = block_matrix(QQ, [[Mz, Z], [Z, Mz]], subdivide=False)
    Wr, _ = weyl_element(X, (1, -1))
    I = identity_matrix(QQ, n)
    rep.check("w^2 = -1 and (z w)^3 = +-1 (relations of SL_2(29))", Wr * Wr == -I and (Zr * Wr) ** 3 in (I, -I))
    classes = maximal_order_classes(X)
    rep.value("conjugacy classes of maximal orders (mass formula checked)", len(classes))
    names = {}
    for ci, (M, w) in enumerate(classes):
        for ch in range(2):
            Gp, Lb = X.lattice(M, ch)
            Li = Lb.inverse()
            WL, ZL = Lb * Wr * Li, Lb * Zr * Li
            if all(x in ZZ for x in WL.list()):
                break
        WL, ZL = WL.change_ring(ZZ), ZL.change_ring(ZZ)
        ok = WL * Gp * WL.transpose() == Gp and ZL * Gp * ZL.transpose() == Gp
        even_unimod = Gp.det() == 1 and all(Gp[i, i] % 2 == 0 for i in range(n))
        cent = 2 * len(elements_of_norm(commutant_basis([ZL, WL], n), Gp, 1))
        rep.value(f"class {ci}: generators preserve the Gram matrix, even unimodular, |centralizer of SL_2(29)|",
                  [bool(ok), bool(even_unimod), int(cent)])
        rep.check(f"class {ci}: a valid even unimodular lattice with SL_2(29)", ok and even_unimod)
        name = {24: "M", 12: "Mtilde"}[int(cent)]
        names[name] = ci
        write_mats(os.path.join(out, f"{name}_gram.txt"),
                   f"Gram matrix of Nebe's lattice L_(56,2)({name}) built from [Neb98] (code/supplementary/nebe/lattices.py)", [Gp])
        write_mats(os.path.join(out, f"{name}_generators.txt"),
                   "z (multiplication by zeta_29) and w (Weyl element) generating SL_2(29); row convention, g G g^T = G", [ZL, WL])
        write_mats(os.path.join(out, f"{name}_order.txt"),
                   "the maximal order: 8 x 8 rational basis matrix in the Q-basis eta_j, eta_j b of H (cycloquat.py)", [M])
        write_mats(os.path.join(out, f"{name}_basis.txt"),
                   "basis matrix of the lattice (56 x 56 rational) in Q(zeta)^2 = Q^56 (cycloquat.py)", [Lb])
    rep.check("one class with centralizer of order 24 (M) and one with 12 (M~)", sorted(names) == ["M", "Mtilde"])
    rep.finish()


if __name__ == "__main__":
    main()
