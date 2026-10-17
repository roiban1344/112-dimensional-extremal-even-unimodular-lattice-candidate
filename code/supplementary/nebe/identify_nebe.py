"""S11. Identification of L with Nebe's lattice L_{56,2}(M~) built from the definition, without any
external data (paper §7.7; Table D2 [C23]). Not used in the proof of Theorem A.
S11. L と、外部データを使わず定義から作った Nebe の格子 L_{56,2}(M~) の同定（論文 §7.7。表 D2 の [C23]）。定理 A の証明には使わない。

  1. L' = L_{56,2}(M~) and generators z', w' of SL_2(29) on it, built from the definition by
     lattices.py (run first; it writes out/nebe/Mtilde_*).
     L' = L_{56,2}(M~) とその上の SL_2(29) の生成元 z', w'。lattices.py が定義から作る（先に走らせる。out/nebe/Mtilde_* を書く）。
  2. An isomorphism of <z', w'> with <g1, g2> by GAP (modulo 3, faithful on finite groups of integer
     matrices); the images are written as words in g1, g2 and recomputed as integer matrices. Both
     classes of the outer automorphism are tried.
     GAP で <z', w'> と <g1, g2> の同型を求める（3 を法として。整数行列の有限群では忠実）。像を g1, g2 の語で書き、整数行列として計算し直す。
     外部自己同型の 2 つの類を両方試す。
  3. The equivariant maps T (z' T = T z, w' T = T w; an 8-dimensional space) are computed exactly, and
     their integral part H (rank 8) is taken.
     同変写像 T（z' T = T z、w' T = T w。8 次元）を厳密に求め、整数の部分 H（階数 8）を取る。
  4. On H the form Q(T) = tr(G'^{-1} T G T^T) is positive definite and equals 56 on isometries; the
     elements with Q <= 56 are enumerated and T G T^T = G' is checked in integers.
     H 上の Q(T) = tr(G'^{-1} T G T^T) は正定値で等長写像では 56。Q ≤ 56 の元を列挙し、T G T^T = G' を整数で確かめる。
  5. tau is transported to L' and identified as right multiplication by an element of M~ of reduced
     norm 5 (and reduced trace 0).
     tau を L' に移し、M~ の被約ノルム 5（被約トレース 0）の元の右乗法として名指しする。
Outputs in out/nebe/: the Gram matrix of L', z', w', the isometry T and tau' = T tau T^{-1}.
`--check` re-checks these files in a few seconds.
出力（out/nebe/）: L' の Gram 行列、z', w'、等長写像 T、tau' = T tau T^{-1}。`--check` はこれらのファイルを数秒で検算し直す。

Requires SageMath (PARI, GAP). Run time: about 45 minutes. Memory: about 3 GB.
SageMath（PARI, GAP）が要る。実行時間: 約 45 分。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/nebe/identify_nebe.py [--check]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))
sys.path.insert(0, os.path.join(HERE, "..", "automorphisms"))
from r112 import data  # noqa: E402
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


def read_mats(path, n):
    from sage.all import ZZ, matrix
    flat = [int(t) for line in open(path) if not line.startswith("#") for t in line.split()]
    return [matrix(ZZ, n, n, flat[k * n * n:(k + 1) * n * n]) for k in range(len(flat) // (n * n))]


def check(rep, out, a):
    """Re-check the output files (seconds). / 出力ファイルを検算し直す（数秒）。"""
    from sage.all import ZZ, identity_matrix, matrix
    n = 56
    G = matrix(ZZ, data.gram(a.data).tolist())
    tau = matrix(ZZ, data.tau(a.data).tolist())
    Gp = read_mats(os.path.join(out, "L56_from_definition_gram.txt"), n)[0]
    zp, wp = read_mats(os.path.join(out, "sl2_29_from_definition_generators.txt"), n)
    T = read_mats(os.path.join(out, "isometry_T.txt"), n)[0]
    taup = read_mats(os.path.join(out, "tau_in_L56_from_definition.txt"), n)[0]
    rep.check("T is unimodular and T G T^T = G'", abs(T.det()) == 1 and T * G * T.transpose() == Gp)
    ok = True
    for h in (zp, wp):
        moved = T.inverse() * h * T
        ok &= h * Gp * h.transpose() == Gp and all(x in ZZ for x in moved.list())
        if ok:
            moved = moved.change_ring(ZZ)
            ok &= moved * G * moved.transpose() == G and moved * tau == tau * moved
    rep.check("z', w' preserve G'; T^{-1} z' T, T^{-1} w' T are isometries of L commuting with tau", ok)
    rep.check("T tau = tau' T and tau'^2 = -5", T * tau == taup * T and taup * taup == -5 * identity_matrix(ZZ, n))


def main():
    from sage.all import GF, QQ, ZZ, block_matrix, identity_matrix, lcm, libgap, matrix, pari, vector, zero_matrix
    ap = argparse.ArgumentParser(description="S11: identification with L_{56,2}(M~)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s11_identify_nebe" + ("_check" if a.check else ""), a.out)
    out = os.path.join(rep.out_dir, "nebe")
    os.makedirs(out, exist_ok=True)
    if a.check:
        check(rep, out, a)
        rep.finish()
        return
    from cycloquat import CycloQuat
    from p1_centralizer import commutant_basis, elements_of_norm

    # 1. L' = L_{56,2}(M~) from the definition, built by lattices.py (run it first).
    # 1. 定義から作った L' = L_{56,2}(M~)（先に lattices.py を走らせる）。
    from lattices import load, read_mats as read_q
    X = CycloQuat(29)
    n = 56
    Gp, (ZL, WL) = load("Mtilde", rep.out_dir)
    M = read_q(os.path.join(out, "Mtilde_order.txt"), 8, QQ)[0]
    Lb = read_q(os.path.join(out, "Mtilde_basis.txt"), 56, QQ)[0]
    rep.check("L' (out/nebe/Mtilde_*) is even unimodular and z', w' preserve its Gram matrix",
              Gp.det() == 1 and all(Gp[i, i] % 2 == 0 for i in range(n)) and all(h * Gp * h.transpose() == Gp for h in (ZL, WL)))
    Cp = commutant_basis([ZL, WL], n)
    rep.check("the centralizer of SL_2(29) in Aut(L') has order 12", 2 * len(elements_of_norm(Cp, Gp, 1)) == 12)

    # 2. Isomorphism of the two copies of SL_2(29). / 2 つの SL_2(29) の同型。
    G = matrix(ZZ, data.gram(a.data).tolist())
    g1, g2 = [matrix(ZZ, g.tolist()) for g in data.generators(a.data)]
    tau = matrix(ZZ, data.tau(a.data).tolist())
    from isometry import equivariant_isometries
    found_isos = equivariant_isometries(Gp, ZL, WL, G, g1, g2, rep)
    result = (found_isos[0][0], found_isos[0][1], found_isos[0][2], len(found_isos)) if found_isos else None
    rep.check("L is SL_2(29)-equivariantly isometric to L_{56,2}(M~)", result is not None)
    T, zi, wi, niso = result
    rep.check("T is unimodular, T G T^T = G', and T intertwines the actions",
              abs(T.det()) == 1 and T * G * T.transpose() == Gp and ZL * T == T * zi and WL * T == T * wi)
    rep.value("equivariant isometries up to sign (expected |centralizer| / 2 = 6)", niso)

    # 5. tau in L'. / L' での tau。
    tp = T * tau * T.inverse()
    rep.check("tau' = T tau T^{-1} is integral, a similitude of norm 5 commuting with z', w'",
              all(x in ZZ for x in tp.list()) and tp * Gp * tp.transpose() == 5 * Gp and tp * ZL == ZL * tp and tp * WL == WL * tp)
    tp = tp.change_ring(ZZ)
    mul = X._mulmat
    Rs = []
    for r in M.rows():
        ux, uy = X.helt(r)
        R = block_matrix(QQ, [[mul(ux), mul(uy)], [-mul(X.conj(uy)), mul(X.conj(ux))]], subdivide=False)
        Rs.append(Lb * R * Lb.inverse())
    coef = matrix(QQ, [R.list() for R in Rs]).solve_left(vector(QQ, tp.list()))
    hq = sum(c * r for c, r in zip(coef, M.rows()))
    rep.value("tau' as right multiplication: coordinates in the basis of M~", [str(c) for c in coef])
    rep.check("tau' is right multiplication by an element of M~ of reduced norm 5", all(c in ZZ for c in coef) and X.nrd(hq) == (5, 0))
    write_mats(os.path.join(out, "L56_from_definition_gram.txt"), "Gram matrix G' of L' = L_{56,2}(M~) built from [Neb98] (code/supplementary/nebe)", [Gp])
    write_mats(os.path.join(out, "sl2_29_from_definition_generators.txt"), "z', w' generating SL_2(29) on L' (row convention, g G' g^T = G')", [ZL, WL])
    write_mats(os.path.join(out, "isometry_T.txt"), "T: rows = images of the basis of L' in the basis of L; T G T^T = G'", [T])
    write_mats(os.path.join(out, "tau_in_L56_from_definition.txt"), "tau' = T tau T^{-1} on L', tau'^2 = -5", [tp])
    check(rep, out, a)
    rep.finish()


if __name__ == "__main__":
    main()
