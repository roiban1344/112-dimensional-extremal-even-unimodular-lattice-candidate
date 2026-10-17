"""S12. Frame-change consistency test (paper §7.2; Table D2 [C17]). Not used in the proof of Theorem A.
S12. 枠の取り替えによる整合性の検査（論文 §7.2。表 D2 の [C17]）。定理 A の証明には使わない。

R = L tensor_{M'} Lambda with Lambda = {(a, b) in M'^2 : b = a tau (mod 2M')} / sqrt2 and the M'-valued
Hermitian form h(lambda, lambda') = (1/2)(a a'* + b b'*). A frame of Lambda is a pair (w1, w2) with
h(w1, w1) = h(w2, w2) = 2 and h(w1, w2) = 0. For every frame, R contains L w1 + L w2 = sqrt2 L + sqrt2 L
(index 2^56), so R is a gluing of two copies of L along some map phi with respect to that frame:
    lambda -> (p, q) = (h(lambda, w1), h(lambda, w2)),  q = p phi (mod 2M').
Since it is the same lattice R, the class of phi (under C \\ . / C and inversion, as in Proposition 3.3)
must be the class of tau for every frame. Done for tau (gluing C) and for tau_B (gluing B).
R = L ⊗_{M'} Lambda、Lambda = {(a, b) ∈ M'^2 : b ≡ a tau (mod 2M')}/√2、M' に値をとるエルミート形式 h(λ, λ') = (1/2)(a a'* + b b'*)。
Lambda の枠とは、h(w1,w1) = h(w2,w2) = 2、h(w1,w2) = 0 の組 (w1, w2)。どの枠についても R ⊃ L w1 ⊕ L w2 ≅ √2L ⊕ √2L（指数 2^56）なので、
R はその枠に関する L 2 枚の貼り合わせで、貼り方の写像 phi は上の式で決まる。同じ格子 R なので、phi の類（命題 3.3 と同じく C の両側からの積と
逆元による類）はどの枠でも tau の類でなければならない。tau（貼り方 C）と tau_B（貼り方 B）について行う。

The elements of norm 2 of Lambda are found as the vectors of norm <= 8 of a 16-dimensional integral
form (qfminim, checked complete against the exact enumerator).
Lambda のノルム 2 の元は、16 次元の整数形式のノルム 8 以下のベクトルとして求める（qfminim。厳密な列挙と照合して完全性を確かめる）。

Requires SageMath. Run time: about 2 minutes. / SageMath が要る。実行時間: 約 2 分。
Run / 実行:  sage -python code/supplementary/frames.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "produce"))
from exact_qfminim import qfminim_checked  # noqa: E402
from p1_centralizer import elements_of_norm  # noqa: E402
from r112 import certs, data  # noqa: E402
from r112.report import Report  # noqa: E402


def main():
    from sage.all import GF, QQ, ZZ, identity_matrix, matrix
    ap = argparse.ArgumentParser(description="S12: frame-change consistency")
    ap.add_argument("--data", default=None)
    ap.add_argument("--certificates", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = Report("s12_frames", a.out)
    n = 56
    I = identity_matrix(ZZ, n)
    G = matrix(ZZ, data.gram(a.data).tolist())
    Ginv = G.inverse()
    adj = lambda Y: G * Y.transpose() * Ginv                      # noqa: E731  (adjoint: <x Y, y> = <x, y Y*>)
    Cb = [matrix(ZZ, X.tolist()) for X in certs.load("commutant.npz", "basis", a.certificates)]
    units = [s * u for u in elements_of_norm(Cb, G, 1) for s in (1, -1)]
    F2 = GF(2)
    U2 = [u.change_ring(F2) for u in units]
    taus = {"C": matrix(ZZ, data.tau(a.data).tolist()),
            "B": matrix(ZZ, certs.load("negative_control_B.npz", "tau_class_B", a.certificates).tolist())}
    cls = {}
    for name, X in taus.items():
        X2 = X.change_ring(F2)
        Xi = X2.inverse()
        cls[name] = {tuple((u * Y * w).list()) for u in U2 for w in U2 for Y in (X2, Xi)}
    rep.value("sizes of the classes of tau (C) and tau_B (B)", {k: len(v) for k, v in cls.items()})
    rep.check("the two classes are different", not (cls["C"] & cls["B"]))
    for name, X in taus.items():
        # Z-basis of Lambda: (c, c tau) and (0, 2c) for c in the basis of M'. / Lambda の Z-基底。
        basis = [(c, c * X) for c in Cb] + [(0 * I, 2 * c) for c in Cb]
        trace_form = lambda u, w: ((u[0] * adj(w[0])).trace() + (u[1] * adj(w[1])).trace()) / n   # noqa: E731
        G16 = matrix(QQ, 16, 16, lambda i, j: 2 * trace_form(basis[i], basis[j])).change_ring(ZZ)
        W = []
        for c in qfminim_checked(G16, 8):
            for sg in (1, -1):
                a0 = sum(sg * ci * b[0] for ci, b in zip(c, basis))
                b0 = sum(sg * ci * b[1] for ci, b in zip(c, basis))
                if a0 * adj(a0) + b0 * adj(b0) == 4 * I:               # h(w, w) = 2
                    W.append((a0, b0))
        frames = [(i, j) for i in range(len(W)) for j in range(len(W))
                  if i != j and W[i][0] * adj(W[j][0]) + W[i][1] * adj(W[j][1]) == 0]
        stat = {}
        for i, j in frames:
            (a1, b1), (a2, b2) = W[i], W[j]
            # Images of e1 = (1, tau), e2 = (0, 2): p = h(e, w1), q = h(e, w2). / e1, e2 の像。
            p1, q1 = (adj(a1) + X * adj(b1)) / 2, (adj(a2) + X * adj(b2)) / 2
            p2, q2 = adj(b1), adj(b2)
            assert all(c in ZZ for M_ in (p1, q1) for c in M_.list()), "h is not M'-valued on Lambda"
            p1b, q1b, p2b, q2b = (M_.change_ring(F2) for M_ in (p1, q1, p2, q2))
            if p1b.is_invertible():
                phi = p1b.inverse() * q1b
                assert p2b * phi == q2b
            elif p2b.is_invertible():
                phi = p2b.inverse() * q2b
                assert p1b * phi == q1b
            else:
                stat["not a graph"] = stat.get("not a graph", 0) + 1
                continue
            k = tuple(phi.list())
            lab = next((f"class {nm}" for nm, v in cls.items() if k in v), "other")
            stat[lab] = stat.get(lab, 0) + 1
        rep.value(f"tau of gluing {name}: elements of norm 2 in Lambda, ordered frames, class seen from each frame",
                  [len(W), len(frames), stat])
        rep.check(f"gluing {name}: every frame gives the class of {name}", stat == {f"class {name}": len(frames)} and len(frames) > 0)
    rep.finish()


if __name__ == "__main__":
    main()
