"""S19. Gluings of two copies of Nebe's L_{56,2}(M) (paper §3.2, §3.3, Proposition 3.3, Table 6 rows for
L_{56,2}(M), Table 10 row "gluing A"; Table D2 [C3], [C4]). Not used in the proof of Theorem A.
S19. Nebe の L_{56,2}(M) 2 枚の貼り方（論文 §3.2、§3.3、命題 3.3、表 6 の L_{56,2}(M) の行、表 10 の「貼り方 A」の行。表 D2 の [C3], [C4]）。
定理 A の証明には使わない。

Uses out/nebe/M_* (lattices.py) and out/nebe/M_tables.npz (tables.py).
out/nebe/M_*（lattices.py）と out/nebe/M_tables.npz（tables.py）を使う。

  (a) the F_2-linear maps of L/2L commuting with SL_2(29) form an 8-dimensional space (the image of M'/2M');
      60 of them are invertible and parity-preserving; under f -> u f u' (u, u' in C) and inversion they
      form 2 classes, of sizes 12 (containing C: forced collisions) and 48 (gluing A);
      SL_2(29) と可換な L/2L の F_2-線形写像は 8 次元（M'/2M' の像）。そのうち 60 個が可逆で偶奇を保ち、u f u'（u, u' ∈ C）と逆元で
      2 類（大きさ 12: C を含む＝強制された衝突、48: 貼り方 A）に分かれる;
  (b) all 96 elements of M' of norm q = 3 are congruent modulo 2 to elements of C (so every gluing from an
      imaginary-quadratic structure of norm 3, e.g. over Q(sqrt -11) or Q(sqrt -2), has forced collisions);
      M' のノルム q = 3 の元 96 個はすべて 2 を法として C の元と合同（ノルム 3 の虚二次構造、たとえば Q(√-11) や Q(√-2) 上の貼り方は
      すべて強制された衝突をもつ）;
  (c) gluing A (an integral lift X of an element f of the class of size 48): L_f is even unimodular;
      every class r_j f + 2L and r_j f^{-1} + 2L has an odd bounding representative (no vector of norm 6,
      and no norm-8 vector of types (6,10), (10,6)); the canonical-form test finds the colliding pairs
      (x_i u_k, f) with norm-8 vectors in the class; their x form 6 regular G-orbits, so L_f has minimum 8
      and exactly 2 * 6 * 24,360 = 292,320 vectors of norm 8; an explicit vector of norm 8 is checked;
      貼り方 A（大きさ 48 の類の元 f の整数の持ち上げ X）: L_f は偶ユニモジュラー。すべての類 r_j f + 2L、r_j f^{-1} + 2L に奇の下界代表元があり
      （ノルム 6 のベクトルも、型 (6,10)・(10,6) のノルム 8 のベクトルも無い）、標準形の検査で衝突する組を求めると、その x は 6 個の正則な G-軌道をなす。
      よって L_f の最小は 8 で、ノルム 8 のベクトルはちょうど 292,320 本。ノルム 8 のベクトルを 1 本、具体的に確かめる;
  (d) the inner-product distributions of the 292,320 norm-8 vectors, to be compared with those of gluing B
      (supplementary/gluing_b.py): identical distributions do not prove that A and B are isometric.
      292,320 本のノルム 8 のベクトルの内積の分布。貼り方 B（supplementary/gluing_b.py）と比べるため。分布が同じでも A と B が等長とは言えない。

Requires SageMath. Run time: about 30 minutes. Memory: about 3 GB.
SageMath が要る。実行時間: 約 30 分。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/nebe/gluings_m.py
"""
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))
from r112 import exact, group, mod2, shells  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import GF, ZZ, block_matrix, identity_matrix, matrix
    from descent import Descent
    from lattices import load
    from p1_centralizer import commutant_basis, elements_of_norm
    ap = argparse.ArgumentParser(description="S19: gluings of two copies of L_{56,2}(M)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s19_nebe_gluings", a.out)
    out = os.path.join(rep.out_dir, "nebe")
    n = 56
    F2 = GF(2)
    Gs, gens_s = load("M", rep.out_dir)
    with np.load(os.path.join(out, "M_tables.npz"), allow_pickle=False) as z:
        U, R6, X8, K8 = z["units"].astype(np.int64), z["min_reps"].astype(np.int64), z["norm8_reps"].astype(np.int64), z["K8"]
    basis = commutant_basis(gens_s, n)
    half = elements_of_norm(basis, Gs, 1)

    # (a) / (a)
    I2 = identity_matrix(F2, n)
    rows = None
    for g in gens_s:
        g2 = g.change_ring(F2)
        Kx = I2.tensor_product(g2.transpose()) - g2.tensor_product(I2)
        rows = Kx if rows is None else rows.stack(Kx)
    E = rows.right_kernel()
    img = matrix(F2, [Y.change_ring(F2).list() for Y in basis]).row_space()
    rep.check("(a) the commutant on L/2L is 8-dimensional and equals the image of M'/2M'", E.dimension() == 8 and E == img)
    Gb = Gs.change_ring(F2)
    q0 = [int(Gs[i, i]) // 2 % 2 for i in range(n)]

    def parity_preserving(f):
        if f * Gb * f.transpose() != Gb:
            return False
        return all((f.row(i).change_ring(ZZ) * Gs * f.row(i).change_ring(ZZ)) // 2 % 2 == q0[i] for i in range(n))
    Eb = [matrix(F2, n, n, list(v)) for v in E.basis()]
    maps = []
    for coeffs in itertools.product([0, 1], repeat=8):
        f = sum((c * b for c, b in zip(coeffs, Eb)), matrix(F2, n, n))
        if f.is_invertible() and parity_preserving(f):
            f.set_immutable()
            maps.append(f)
    rep.check("(a) exactly 60 invertible parity-preserving maps", len(maps) == 60)
    C2 = []
    for Y in half:
        Yb = Y.change_ring(F2)
        Yb.set_immutable()
        if Yb not in C2:
            C2.append(Yb)
    todo, classes = set(maps), []
    while todo:
        f0 = min(todo, key=lambda f: tuple(f.list()))
        orb = set()
        for u in C2:
            for v in C2:
                for f in (f0, f0.inverse()):
                    h = u * f * v
                    h.set_immutable()
                    orb.add(h)
        classes.append(orb)
        todo -= orb
    info = sorted(((len(o), any(u in o for u in C2)) for o in classes))
    rep.value("(a) classes (size, contains an element of C)", info)
    rep.check("(a) two classes: 12 (containing C) and 48", info == [(12, True), (48, False)])

    # (b) / (b)
    norm3 = [s * Y for Y in elements_of_norm(basis, Gs, 3) for s in (1, -1)]
    Cset = set(C2)
    in_C = 0
    for Y in norm3:
        Yb = Y.change_ring(F2)
        Yb.set_immutable()
        in_C += Yb in Cset
    rep.value("(b) elements of M' of norm 3, and how many are congruent mod 2 to an element of C", [len(norm3), in_C])
    rep.check("(b) all 96 are congruent mod 2 to elements of C", len(norm3) == 96 and in_C == 96)

    # (c) gluing A / 貼り方 A
    clsA = next(o for o in classes if len(o) == 48)
    lift = None
    for N in range(1, 8):                                       # a short integral lift of an element of the class / 類の元の短い整数の持ち上げ
        for Y in elements_of_norm(basis, Gs, N):
            Yb = Y.change_ring(F2)
            Yb.set_immutable()
            if Yb in clsA:
                lift = Y
                break
        if lift is not None:
            break
    rep.check("(c) an integral lift X in M' of an element of the class of size 48", lift is not None)
    Xs = lift
    I = identity_matrix(ZZ, n)
    Bm = block_matrix(ZZ, [[I, Xs], [0 * I, 2 * I]], subdivide=False)
    GA = Bm * block_matrix(ZZ, [[Gs, 0], [0, Gs]], subdivide=False) * Bm.transpose() / 2
    rep.check("(c) L_f is integral, even, of determinant 1",
              all(c in ZZ for c in GA.list()) and all(GA[i, i] % 2 == 0 for i in range(2 * n)) and GA.det() == 1)
    GA = GA.change_ring(ZZ)
    G = np.array(Gs.rows(), dtype=np.int64)
    Ginv = exact.inverse_unimodular(G)
    S = group.sl2_29_mod_sign([np.array(g.rows(), dtype=np.int64) for g in gens_s])
    Xn = np.array(Xs.rows(), dtype=np.int64)
    Xinv = np.array(Xs.change_ring(F2).inverse().change_ring(ZZ).rows(), dtype=np.int64)
    involutive = bool((((Xn @ Xn) - np.eye(n, dtype=np.int64)) % 2 == 0).all())
    rep.value("(c) f^2 = 1 mod 2", involutive)
    d = Descent(G, S, R6)
    starts = [exact.i64_matmul(R6, Xn)] + ([] if involutive else [exact.i64_matmul(R6, Xinv)])
    W = np.concatenate([np.stack([d.descend(w, 40)[0] for w in st]) for st in starts])
    del d
    mv = shells.MinimalVectors(G, R6, S)
    mx = mv.max_abs_inner(W)
    del mv
    nw = exact.norms(W, G)
    rep.check("(c) odd bounding representatives for all classes r_j f + 2L (and r_j f^-1 + 2L): no norm 6, no (6,10), (10,6)",
              np.isin(nw, (14, 18)).all() and (mx <= 6).all())
    can = mod2.Canonical(S)
    XU = np.concatenate([exact.i64_matmul(X8, u)[:, None, :] for u in U], axis=1).reshape(-1, n)
    hits = []
    for i in range(0, len(XU), 1 << 14):
        f = can.canon(mod2.pack(exact.i64_matmul(XU[i:i + (1 << 14)], Xn)))
        hits += (i + np.flatnonzero(np.isin(f, K8))).tolist()
    xf, xs = can.canon_and_orbit_size(mod2.pack(XU[hits]))
    norbits = len(set(xf.tolist()))
    rep.value("(c) pairs tested, colliding pairs, G-orbits of the colliding x", [len(XU), len(hits), norbits])
    rep.check("(c) the colliding x form 6 regular G-orbits: minimum 8 and 292320 vectors of norm 8",
              norbits == 6 and (2 * xs == 24360).all())
    # An explicit norm-8 vector (as in gluing_b.py). / ノルム 8 のベクトルを 1 本（gluing_b.py と同じ）。
    allf, _ = can.canon_and_orbit_size(mod2.pack(XU))
    x = XU[hits[0]]
    c_img = can.images(mod2.pack(exact.i64_matmul(x, Xn)))[0]
    j = int(np.flatnonzero(allf == c_img.min())[0])
    y_img = can.images(mod2.pack(XU[j]))[0]
    v = exact.i64_matmul(exact.i64_matmul(XU[j], S[int(np.argmin(y_img))].astype(np.int64)),
                         group.inverse_isometry(S[int(np.argmin(c_img))].astype(np.int64), G, Ginv))
    yv = (v - exact.i64_matmul(x, Xn)) // 2
    zc = np.concatenate([x, yv])
    GAn = np.array(GA.rows(), dtype=np.int64)
    rep.check("(c) an explicit vector of norm 8 in L_f", ((v - exact.i64_matmul(x, Xn)) % 2 == 0).all() and int(zc @ GAn @ zc) == 8)

    # (d) inner-product distributions / 内積の分布
    reps_xv = {}
    for h in hits:
        xx = XU[h]
        ci = can.images(mod2.pack(exact.i64_matmul(xx, Xn)))[0]
        jj = int(np.flatnonzero(allf == ci.min())[0])
        yi = can.images(mod2.pack(XU[jj]))[0]
        vv = exact.i64_matmul(exact.i64_matmul(XU[jj], S[int(np.argmin(yi))].astype(np.int64)),
                              group.inverse_isometry(S[int(np.argmin(ci))].astype(np.int64), G, Ginv))
        reps_xv.setdefault(int(can.canon(mod2.pack(xx))[0]), (xx, vv))
    S64 = S.astype(np.int64)
    XS, VS = [], []
    for xx, vv in reps_xv.values():
        xg, vg = np.einsum("j,gjk->gk", xx, S64), np.einsum("j,gjk->gk", vv, S64)
        for sx in (1, -1):
            for sv in (1, -1):
                XS.append(sx * xg)
                VS.append(sv * vg)
    XS, VS = np.concatenate(XS), np.concatenate(VS)
    rep.check("(d) 292320 distinct norm-8 vectors", len(np.unique(np.concatenate([XS, VS], axis=1), axis=0)) == 292320)
    XG, VG = exact.f64_matmul(XS, G).astype(np.float64), exact.f64_matmul(VS, G).astype(np.float64)
    dists = {}
    for i0 in range(0, len(XS), len(S)):
        ip = np.rint((XG @ XS[i0] + VG @ VS[i0]) / 2).astype(np.int64)
        vals, cnt = np.unique(ip, return_counts=True)
        key = str(list(zip(vals.tolist(), cnt.tolist())))
        dists[key] = dists.get(key, 0) + 1
    rep.value("(d) inner-product distributions of the norm-8 vectors of gluing A: {distribution: blocks}", dists)
    rep.finish()


if __name__ == "__main__":
    main()
