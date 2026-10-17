"""S20. Reduction of F1 to Nebe's lattice: "min L_{56,2}(M) = 6 implies min L = 6" (paper §6.1 (b);
Table D2 [C6]). A second route to F1, not used in the proof of Theorem A (which uses verify/v1).
S20. F1 の Nebe の格子への帰着: 「min L_{56,2}(M) = 6 ならば min L = 6」（論文 §6.1 (b)。表 D2 の [C6]）。F1 への 2 本目の道で、定理 A の証明
（verify/v1 を使う）には使わない。

  1. For the elements X of M'(L_{56,2}(M)) whose reduction modulo 2 has rank 28 (found among the elements
     of small norm q), the sublattices L_X = L_{56,2}(M) X + 2 L_{56,2}(M), rescaled by 1/sqrt2, are formed;
     they are SL_2(29)-invariant. One of them is shown to be SL_2(29)-equivariantly isometric to L
     (isometry.py), by an explicit integer matrix.
     M'(L_{56,2}(M)) の元 X で 2 を法とした階数が 28 のもの（ノルム q の小さな元から探す）について、部分格子 L_X = L_{56,2}(M) X + 2 L_{56,2}(M)
     を 1/√2 倍したものを作る（SL_2(29)-不変）。そのうち 1 つが L と SL_2(29) 同変に等長であることを、整数行列で示す（isometry.py）。
  2. Then norm-2 vectors of L correspond to norm-4 vectors of L_{56,2}(M), and norm-4 vectors of L to
     norm-8 vectors of L_{56,2}(M) whose class lies in the 28-dimensional SL_2(29)-invariant subspace
     W = (L_{56,2}(M) X + 2 L_{56,2}(M)) / 2 L_{56,2}(M).
     すると L のノルム 2 のベクトルは L_{56,2}(M) のノルム 4 のベクトルに、L のノルム 4 のベクトルは、類が 28 次元の SL_2(29)-不変部分空間 W に入る
     L_{56,2}(M) のノルム 8 のベクトルに対応する。
  3. Using the complete table of norm-8 orbits of L_{56,2}(M) (tables.py): no pair (x_i, u_k) has its class in W,
     and W is SL_2(29)-invariant, so no norm-8 vector of L_{56,2}(M) has its class in W. Positive control:
     the 656 minimal-vector representatives of L, mapped into L_{56,2}(M), have norm 12 and class in W.
     L_{56,2}(M) のノルム 8 の軌道の完全な表（tables.py）を使う: どの組 (x_i, u_k) も類が W に入らず、W は SL_2(29)-不変なので、L_{56,2}(M) の
     ノルム 8 のベクトルで類が W に入るものは無い。陽性対照: L の最小ベクトルの代表 656 本を L_{56,2}(M) に移すと、ノルム 12 で類は W に入る。
Hence min L_{56,2}(M) = 6 (proved in tables.py, and reported in [Neb98, Table II]) implies min L = 6.
よって min L_{56,2}(M) = 6（tables.py で証明。[Neb98, Table II] にも報告がある）ならば min L = 6。

Requires SageMath. Run time: about 1 hour per candidate sublattice (the isometry search). Memory: about 3 GB.
SageMath が要る。実行時間: 候補の部分格子 1 つにつき約 1 時間（等長写像の探索）。メモリ: 約 3 GB。
Run / 実行:  sage -python code/supplementary/nebe/reduction_to_nebe.py
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))
from r112 import certs, data, exact  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import GF, QQ, ZZ, identity_matrix, matrix
    from isometry import equivariant_isometries
    from lattices import load, write_mats
    from p1_centralizer import commutant_basis, elements_of_norm
    ap = argparse.ArgumentParser(description="S20: reduction of F1 to L_{56,2}(M)")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s20_reduction_to_nebe", a.out)
    out = os.path.join(rep.out_dir, "nebe")
    n = 56
    F2 = GF(2)
    I = identity_matrix(ZZ, n)
    GM, (zM, wM) = load("M", rep.out_dir)
    G = matrix(ZZ, data.gram(a.data).tolist())
    g1, g2 = [matrix(ZZ, g.tolist()) for g in data.generators(a.data)]
    basis = commutant_basis([zM, wM], n)

    # 1. candidate sublattices / 候補の部分格子
    spaces = {}
    for N in range(1, 7):
        for X in elements_of_norm(basis, GM, N):
            X2 = X.change_ring(F2)
            if X2.rank() == n // 2:
                spaces.setdefault(tuple(X2.row_space().basis_matrix().list()), X)
    rep.value("SL_2(29)-invariant subspaces of dimension 28 found from elements of norm <= 6", len(spaces))
    found = None
    for k, X in enumerate(spaces.values()):
        Bm = X.stack(2 * I).hermite_form(include_zero_rows=False)
        GX = Bm * GM * Bm.transpose() / 2
        if not (all(c in ZZ for c in GX.list()) and GX.det() == 1 and all(GX[i, i] % 2 == 0 for i in range(n))):
            rep.value(f"subspace {k}: the rescaled sublattice is not even unimodular", True)
            continue
        GX = GX.change_ring(ZZ)
        Bi = Bm.change_ring(QQ).inverse()
        hz, hw = (Bm * h * Bi for h in (zM, wM))
        assert all(c in ZZ for h in (hz, hw) for c in h.list())
        hz, hw = hz.change_ring(ZZ), hw.change_ring(ZZ)
        isos = equivariant_isometries(GX, hz, hw, G, g1, g2, rep)
        rep.value(f"subspace {k}: SL_2(29)-equivariant isometries to L (up to sign)", len(isos))
        if isos:
            found = (X, Bm, isos[0][0])
            break
    rep.check("1. one sublattice L_X is SL_2(29)-equivariantly isometric to L", found is not None)
    X, Bm, T = found                                   # T G T^T = G_X: rows of T = basis of L_X in the basis of L
    Ti = T.inverse()
    assert all(c in ZZ for c in Ti.list())
    Emb = (Ti * Bm).change_ring(ZZ)                    # rows: the basis of L, in coordinates of L_{56,2}(M)
    rep.check("1. the embedding e: L -> L_{56,2}(M) satisfies <e(x), e(y)> = 2 <x, y>", Emb * GM * Emb.transpose() == 2 * G)
    write_mats(os.path.join(out, "embedding_L_into_M.txt"),
               "rows: images of the basis of L in L_{56,2}(M), scaled by sqrt2 (Emb G_M Emb^T = 2 G)", [Emb])

    # 2-3. the subspace W and the table / 部分空間 W と表
    W = Emb.change_ring(F2).row_space()
    rep.check("2. W has dimension 28 and is SL_2(29)-invariant",
              W.dimension() == 28 and all(w * h.change_ring(F2) in W for w in W.basis() for h in (zM, wM)))
    P = np.array(W.basis_matrix().right_kernel_matrix().transpose().change_ring(ZZ), dtype=np.int64)   # v in W iff v P = 0 mod 2

    def in_W(V):
        return ~((((np.asarray(V) & 1) @ P) & 1).any(axis=1))
    with np.load(os.path.join(out, "M_tables.npz"), allow_pickle=False) as z:
        U, X8 = z["units"].astype(np.int64), z["norm8_reps"].astype(np.int64)
    hits = sum(int(in_W(exact.i64_matmul(X8, u)).sum()) for u in U)
    rep.value("3. pairs (x_i, u_k) of L_{56,2}(M) tested, classes in W", [len(X8) * len(U), hits])
    rep.check("3. no norm-8 vector of L_{56,2}(M) has its class in W", hits == 0)
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    En = np.array(Emb.rows(), dtype=np.int64)
    Y = exact.i64_matmul(R6, En)
    rep.check("3. positive control: the 656 minimal vectors of L go to norm 12 in L_{56,2}(M), with class in W",
              (exact.norms(Y, np.array(GM.rows(), dtype=np.int64)) == 12).all() and in_W(Y).all())
    rep.finish()


if __name__ == "__main__":
    main()
