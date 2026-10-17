"""SL_2(29)-equivariant isometries between two 56-dimensional lattices with an action of SL_2(29)
(used by identify_nebe.py and reduction_to_nebe.py). Requires SageMath (GAP, PARI).
SL_2(29) が作用する 2 つの 56 次元格子の間の、SL_2(29) 同変な等長写像（identify_nebe.py と reduction_to_nebe.py が使う）。SageMath（GAP, PARI）が要る。

Given lattices L' (Gram G', generators z', w') and L (Gram G, generators g1, g2):
  1. an isomorphism <z', w'> -> <g1, g2> is found by GAP on the reductions modulo 3 (faithful on finite
     groups of integer matrices); images are written as words in g1, g2 and recomputed as integer matrices;
     both classes of the outer automorphism of SL_2(29) are tried;
  2. the equivariant maps T (z' T = T phi(z'), w' T = T phi(w')) are computed exactly (an 8-dimensional
     space), and their integral part H is taken (saturation);
  3. on H, Q(T) = tr(G'^{-1} T G T^T) is positive definite and equals 56 on isometries; all T with
     Q <= 56 are enumerated (qfminim, checked complete against the exact enumerator) and T G T^T = G' is
     checked in integers.
L'（Gram G'、生成元 z', w'）と L（Gram G、生成元 g1, g2）について:
  1. GAP で <z', w'> -> <g1, g2> の同型を 3 を法とした還元の上で求める（整数行列の有限群では忠実）。像は g1, g2 の語で書いて整数行列として
     計算し直す。SL_2(29) の外部自己同型の 2 つの類を両方試す;
  2. 同変写像 T（8 次元の空間）を厳密に求め、その整数の部分 H（飽和）を取る;
  3. H 上の Q(T) = tr(G'^{-1} T G T^T) は正定値で等長写像では 56。Q ≤ 56 の T をすべて列挙し（qfminim。厳密な列挙と照合して完全性を確かめる）、
     T G T^T = G' を整数で確かめる。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "automorphisms"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))


def equivariant_isometries(Gp, zp, wp, G, g1, g2, rep, n=56):
    """List of (T, phi(z'), phi(w')) for the first outer class admitting isometries, or [] if none.
    T: integer matrix with T G T^T = G' and z' T = T phi(z'), w' T = T phi(w') (rows of T = images of the basis
    of L' in the basis of L). Returns all such T up to sign for that class.
    等長写像が存在する最初の外部類について (T, phi(z'), phi(w')) のリスト（無ければ []）。符号を除いてすべての T を返す。"""
    from common import intertwiners
    from exact_qfminim import qfminim_checked
    from sage.all import GF, QQ, ZZ, identity_matrix, lcm, libgap, matrix
    F3 = GF(3)
    GA = libgap.Group([libgap(zp.change_ring(F3)), libgap(wp.change_ring(F3))])
    GB = libgap.Group([libgap(g1.change_ring(F3)), libgap(g2.change_ring(F3))])
    rep.check("both groups have order 24360", int(GA.Size()) == int(GB.Size()) == 24360)
    iso = libgap.IsomorphismGroups(GA, GB)
    rep.check("GAP finds an isomorphism", iso != libgap.fail)
    autA = libgap.AutomorphismGroup(GA)
    tr = libgap.RightTransversal(autA, libgap.InnerAutomorphismsAutomorphismGroup(autA))
    outreps = [tr[k] for k in range(int(libgap.Size(tr)))]
    hom = libgap.EpimorphismFromFreeGroup(GB)
    gg, gi = {1: g1, 2: g2}, {1: g1.inverse(), 2: g2.inverse()}

    def lift(img):
        word = libgap.LetterRepAssocWord(libgap.PreImagesRepresentative(hom, img)).sage()
        A = identity_matrix(ZZ, n)
        for e in word:
            A = A * (gg[e] if e > 0 else gi[-e])
        return A.change_ring(ZZ)

    Gpi = Gp.inverse()
    for k, alpha in enumerate(outreps):
        phi = libgap.CompositionMapping(iso, alpha)
        zi = lift(libgap.Image(phi, libgap(zp.change_ring(F3))))
        wi = lift(libgap.Image(phi, libgap(wp.change_ring(F3))))
        V = intertwiners([zp, wp], [zi, wi], n)
        rep.value(f"outer class {k}: dimension of the equivariant maps L' -> L", len(V))
        if not V:
            continue
        B = matrix(QQ, [T.list() for T in V])
        B = matrix(ZZ, [r * lcm([x.denominator() for x in r]) for r in B.rows()])
        Hb = [matrix(ZZ, n, n, list(r)) for r in B.saturation().rows()]
        Q = matrix(QQ, len(Hb), len(Hb), lambda i, j: (Gpi * Hb[i] * G * Hb[j].transpose()).trace())
        Q = (Q + Q.transpose()) / 2
        assert Q.is_positive_definite()
        dQ = lcm([x.denominator() for x in Q.list()])
        cands = qfminim_checked((dQ * Q).change_ring(ZZ), dQ * n)
        isos = [T for T in (sum(c_ * H_ for c_, H_ in zip(c, Hb)) for c in cands) if T * G * T.transpose() == Gp]
        rep.value(f"outer class {k}: integral equivariant maps with Q <= 56, isometries (up to sign)", [len(cands), len(isos)])
        if isos:
            return [(T, zi, wi) for T in isos]
    return []
