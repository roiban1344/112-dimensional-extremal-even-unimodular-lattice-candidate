"""S13. The negative control B: min R_B = 8 and exactly 292,320 vectors of norm 8 (paper Theorem A,
Table 6, Table 10, §6.5; Table D2 [C4]). Not used in the proof of Theorem A for R.
S13. 陰性対照 B: min R_B = 8 で、ノルム 8 のベクトルはちょうど 292,320 本（論文の定理 A、表 6、表 10、§6.5。表 D2 の [C4]）。
R についての定理 A の証明には使わない。

tau_B (certificates/negative_control_B.npz, key tau_class_B) is an element of M' with tau_B^2 = -5 in the other class.
The same arguments as for R, with tau replaced by tau_B / 同じ議論を tau の代わりに tau_B で行う:
  (a) R_B is even unimodular, and SL_2(29) commutes with tau_B (as V0);
      R_B は偶ユニモジュラーで、SL_2(29) は tau_B と可換（V0 と同じ）。
  (b) types (6,6), (6,10), (10,6): every class r_j tau_B + 2L has an odd bounding representative,
      found by the descent (produce/descent.py) and checked exactly against all minimal vectors (as V5);
      so R_B has no vector of norm 6 and no norm-8 vector of these types;
      型 (6,6), (6,10), (10,6): すべての類 r_j tau_B + 2L に奇の下界代表元がある（降下で見つけ、全最小ベクトルとの内積で厳密に確かめる。V5 と同じ）。
      よって R_B にノルム 6 のベクトルは無く、ノルム 8 のベクトルはこれらの型から来ない。
  (c) type (8,8) by canonical forms (as V6; uses out/K8.npy from V4): exactly the listed pairs collide;
      their x form 6 regular G-orbits. For each colliding pair an explicit partner v of norm 8 with
      v = x tau_B (mod 2L) is constructed and the vector (x, v)/sqrt2 of R_B is checked to have norm 8;
      型 (8,8) を標準形で（V6 と同じ。V4 の out/K8.npy を使う）: 衝突する組を列挙し、その x は 6 個の正則な G-軌道をなす。各組について
      v ≡ x tau_B (mod 2L) となるノルム 8 の相手 v を具体的に作り、R_B のベクトル (x, v)/√2 のノルムが 8 であることを確かめる。
  (d) count: each colliding x has exactly one partner +-v (midpoint lemma), so the norm-8 vectors of R_B
      are (+-x, +-v) over the 6 orbits: 2 * 6 * 24,360 = 292,320. Their inner-product distributions
      (used to compare with gluing A) are recorded.
      数え上げ: 衝突する x の相手は ±v だけ（中点補題）なので、R_B のノルム 8 のベクトルは 6 軌道にわたる (±x, ±v)、2 × 6 × 24,360 = 292,320 本。
      内積の分布（貼り方 A との比較用）も記録する。
The colliding pairs (x, v) are written to out/gluing_b_collisions.npz for supplementary/anatomy_b.py.
衝突する組 (x, v) は supplementary/anatomy_b.py のために out/gluing_b_collisions.npz に書く。

Only numpy. Run time: about 15 minutes (after V4). Memory: about 2.5 GB.
numpy のみ。実行時間: 約 15 分（V4 のあと）。メモリ: 約 2.5 GB。
Run / 実行:  python3 code/supplementary/gluing_b.py
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "produce"))
from descent import Descent  # noqa: E402
from r112 import certs, data, exact, group, mod2, shells  # noqa: E402
from r112.report import Report  # noqa: E402

N8 = 36957286800


def main():
    ap = argparse.ArgumentParser(description="S13: the negative control B")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s13_gluing_b", a.out)
    mm = exact.i64_matmul
    G = data.gram(a.data)
    Ginv = exact.inverse_unimodular(G)
    gens = data.generators(a.data)
    tB = certs.load("negative_control_B.npz", "tau_class_B", a.certificates).astype(np.int64)
    I = np.eye(56, dtype=np.int64)

    # (a)
    rep.check("(a) tau_B^2 = -5 I and tau_B G tau_B^T = 5 G", (mm(tB, tB) == -5 * I).all() and (mm(mm(tB, G), tB.T) == 5 * G).all())
    rep.check("(a) tau_B commutes with g1, g2", all((mm(g, tB) == mm(tB, g)).all() for g in gens))
    GR = data.gram_R(G, tB)
    rep.check("(a) R_B: even, det 1", (np.diag(GR) % 2 == 0).all() and exact.det_bareiss(GR) == 1)
    rep.check("(a) tau_B^2 = 1 (mod 2), so the types (6,10) and (10,6) coincide", ((mm(tB, tB) - I) % 2 == 0).all())

    S = group.sl2_29_mod_sign(gens)
    R6 = certs.load("min_reps.npz", "reps", a.certificates).astype(np.int64)
    X = certs.load("norm8_reps.npz", "reps", a.certificates).astype(np.int64)
    U = certs.load("centralizer.npz", "units", a.certificates).astype(np.int64)

    # (b) odd bounding representatives for tau_B. / tau_B の奇の下界代表元。
    d = Descent(G, S, R6)
    W = np.stack([d.descend(mm(r, tB), 40)[0] for r in R6])
    del d
    norms = exact.norms(W, G)
    rep.check("(b) w0 - r_j tau_B has even coordinates, norms in {14, 18}",
              ((W - mm(R6, tB)) % 2 == 0).all() and np.isin(norms, (14, 18)).all())
    mv = shells.MinimalVectors(G, R6, S)
    mx = mv.max_abs_inner(W)
    del mv
    rep.value("(b) max |<w0, u>| over all minimal vectors: {value: count}", {int(k): int(c) for k, c in zip(*np.unique(mx, return_counts=True))})
    rep.check("(b) all 656 classes r_j tau_B + 2L have an odd bounding representative (no norm 6, no (6,10))", (mx <= 6).all())

    # (c) type (8,8). / 型 (8,8)。
    K8 = np.load(os.path.join(rep.out_dir, "K8.npy"))
    rep.check("K_8 from V4 has 1517130 entries", len(K8) == 1517130)
    can = mod2.Canonical(S)
    XU = np.concatenate([mm(X, u)[:, None, :] for u in U], axis=1).reshape(-1, 56)     # row 6 i + k
    hits = []
    step = 1 << 14
    for i in range(0, len(XU), step):
        f = can.canon(mod2.pack(mm(XU[i:i + step], tB)))
        hits += (i + np.flatnonzero(np.isin(f, K8))).tolist()
    xforms, xsizes = can.canon_and_orbit_size(mod2.pack(XU[hits]))
    orbit_forms = sorted(set(xforms.tolist()))
    rep.value("(c) colliding pairs (row 6 i + k)", hits)
    rep.value("(c) G-orbits of the colliding x, and their sizes", [len(orbit_forms), sorted(set((2 * xsizes).tolist()))])
    rep.check("(c) the colliding x form 6 regular G-orbits", len(orbit_forms) == 6 and (2 * xsizes == 24360).all())
    # Partner v: if can(x tau_B) = can(y) with y = x_j u_l in the table, i.e. (x tau_B) s = y s' (mod 2),
    # then v = y s' s^{-1} is a norm-8 vector in the class x tau_B + 2L.
    # 相手 v: can(x tau_B) = can(y)（y は表の x_j u_l）、すなわち (x tau_B) s ≡ y s' (mod 2) なら、v = y s' s^{-1} は類 x tau_B + 2L のノルム 8 のベクトル。
    allforms, _ = can.canon_and_orbit_size(mod2.pack(XU))
    first = {}
    for idx, f in enumerate(allforms.tolist()):
        first.setdefault(f, idx)
    partners = []
    for h in hits:
        x = XU[h]
        c = mod2.pack(mm(x, tB))
        img_c = can.images(c)[0]
        f = int(img_c.min())
        y = XU[first[f]]
        img_y = can.images(mod2.pack(y))[0]
        s = S[int(np.argmin(img_c))].astype(np.int64)
        s2 = S[int(np.argmin(img_y))].astype(np.int64)
        v = mm(mm(y, s2), group.inverse_isometry(s, G, Ginv))
        assert ((v - mm(x, tB)) % 2 == 0).all() and int(v @ G @ v) == 8
        yv = (v - mm(x, tB)) // 2
        zc = np.concatenate([x, yv])                                     # coordinates in the basis (3.1)
        assert int(zc @ GR @ zc) == 8
        partners.append(v)
    rep.check("(c) every colliding pair has an explicit partner v of norm 8, and (x, v)/sqrt2 has norm 8 in R_B", len(partners) == len(hits))
    # Saved for supplementary/anatomy_b.py. / supplementary/anatomy_b.py のために保存する。
    certs.save_npz(os.path.join(rep.out_dir, "gluing_b_collisions.npz"), x=XU[hits], v=np.stack(partners), tau_B=tB)

    # (d) count and inner-product distributions. / 数え上げと内積の分布。
    rep.value("(d) norm-8 vectors of R_B = 2 * (sum of the orbit sizes of the colliding x)", 2 * 6 * 24360)
    reps_xv = {}
    for h, v in zip(hits, partners):
        reps_xv.setdefault(int(can.canon(mod2.pack(XU[h]))[0]), (XU[h], v))
    S64 = S.astype(np.int64)
    XS, VS = [], []
    for x, v in reps_xv.values():
        xg, vg = np.einsum("j,gjk->gk", x, S64), np.einsum("j,gjk->gk", v, S64)
        for sx in (1, -1):
            for sv in (1, -1):
                XS.append(sx * xg)
                VS.append(sv * vg)
    XS, VS = np.concatenate(XS), np.concatenate(VS)
    keys = np.concatenate([XS, VS], axis=1)
    rep.check("(d) these are 292320 distinct vectors", len(np.unique(keys, axis=0)) == 292320)
    XG, VG = exact.f64_matmul(XS, G).astype(np.float64), exact.f64_matmul(VS, G).astype(np.float64)
    dists = {}
    for i0 in range(0, len(XS), len(S)):
        ip = np.rint((XG @ XS[i0] + VG @ VS[i0]) / 2).astype(np.int64)  # <z0, z> = (1/2)(<x0,x> + <v0,v>)
        vals, cnt = np.unique(ip, return_counts=True)
        key = str(list(zip(vals.tolist(), cnt.tolist())))
        dists[key] = dists.get(key, 0) + 1
    rep.value("(d) inner-product distributions of the norm-8 vectors (one per block of 12180): {distribution: blocks}", dists)
    rep.finish()


if __name__ == "__main__":
    main()
