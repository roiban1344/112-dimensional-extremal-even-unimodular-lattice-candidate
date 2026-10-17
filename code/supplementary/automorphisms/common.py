"""Shared setup for the computations on Aut(R) (paper §7.8, Appendix G). Requires SageMath.
Aut(R) の計算（論文 §7.8、付録 G）の共通の準備。SageMath が要る。

R sits in V = (L tensor Q) + (L tensor Q) with row vectors (x, v) and inner product
(1/2)(<x, x'> + <v, v'>). Its basis matrix is B = [[I, tau], [0, 2I]] (rows (e_i, e_i tau), (0, 2e_i);
the factor 1/sqrt2 is absorbed in the form), and its Gram matrix is G_R = (3.1). A matrix X on V
(acting on ambient row vectors) preserves R iff B X B^{-1} is integral; in the R-basis it is
X_R = B X B^{-1}, and it is an isometry iff X_R G_R X_R^T = G_R.
R は V = (L⊗Q) ⊕ (L⊗Q)（行ベクトル (x, v)、内積 (1/2)(<x,x'> + <v,v'>)）の中にある。基底行列は B = [[I, tau], [0, 2I]]、
Gram 行列は G_R = (3.1)。V 上の行列 X が R を保つ ⟺ B X B^{-1} が整数。R の基底では X_R = B X B^{-1} で、
X_R G_R X_R^T = G_R なら等長変換。

SL_2(29) acts diagonally, g -> diag(g, g). The endomorphisms of V commuting with it are the block
matrices [[a, b], [c, d]] with a, b, c, d in M' tensor Q (a 32-dimensional space, M' = commutant.npz).
SL_2(29) は対角に g -> diag(g, g) で作用する。それと可換な V の自己準同型は、a, b, c, d ∈ M'⊗Q のブロック行列 [[a, b], [c, d]]
（32 次元。M' は commutant.npz）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "produce"))
from exact_qfminim import qfminim_checked  # noqa: E402
from r112 import certs, data  # noqa: E402

N = 56


class Setup:
    """Matrices of the data in Sage, and R in the basis (3.1). / データの Sage 行列と、基底 (3.1) での R。"""

    def __init__(self, data_dir=None, cert_dir=None):
        from sage.all import QQ, ZZ, block_matrix, identity_matrix, matrix, zero_matrix
        n = N
        self.G = matrix(ZZ, data.gram(data_dir).tolist())
        self.g1, self.g2 = [matrix(ZZ, g.tolist()) for g in data.generators(data_dir)]
        self.tau = matrix(ZZ, data.tau(data_dir).tolist())
        self.Mb = [matrix(ZZ, X.tolist()) for X in certs.load("commutant.npz", "basis", cert_dir)]
        assert len(self.Mb) == 8 and all(X * g == g * X for X in self.Mb for g in (self.g1, self.g2))
        I, Z = identity_matrix(ZZ, n), zero_matrix(ZZ, n, n)
        self.B = block_matrix(ZZ, [[I, self.tau], [Z, 2 * I]], subdivide=False).change_ring(QQ)
        self.Bi = self.B.inverse()
        GR = self.B * block_matrix(QQ, [[self.G, 0], [0, self.G]], subdivide=False) / 2 * self.B.transpose()
        assert all(x in ZZ for x in GR.list())
        self.GR = GR.change_ring(ZZ)
        assert self.GR.det() == 1
        self.GRi = self.GR.inverse().change_ring(ZZ)
        # The generators of SL_2(29) acting on R, in the R-basis. / R に作用する SL_2(29) の生成元（R の基底で）。
        self.sR = [self.in_R(block_matrix(QQ, [[g, 0], [0, g]], subdivide=False)) for g in (self.g1, self.g2)]
        self.cert_dir = cert_dir

    def in_R(self, X):
        """Ambient matrix X -> matrix in the R-basis (asserted integral). / 周囲の行列 X → R の基底での行列（整数であることを確かめる）。"""
        from sage.all import ZZ
        Y = self.B * X * self.Bi
        assert all(x in ZZ for x in Y.list()), "does not preserve R"
        return Y.change_ring(ZZ)

    def block_generators(self, Ts):
        """For 8 matrices T on L: the 32 matrices with one block T in position (0,0), (0,1), (1,0) or (1,1),
        in the R-basis (rational). / L 上の 8 個の行列 T から、1 つのブロックだけが T の 32 個の行列（R の基底、有理数）。"""
        from sage.all import QQ, block_matrix, zero_matrix
        Z = zero_matrix(QQ, N, N)
        out = []
        for t in Ts:
            for pos in range(4):
                blocks = [[Z, Z], [Z, Z]]
                blocks[pos // 2][pos % 2] = t.change_ring(QQ)
                out.append(self.B * block_matrix(QQ, blocks, subdivide=False) * self.Bi)
        return out

    def isometries_from(self, Ts):
        """All isometries of R (up to sign) in the Z-lattice H = { sum y_k X_k integral } spanned by the 32
        block matrices built from Ts. On H, Q(X) = tr(G_R^{-1} X G_R X^T) is positive definite and equals
        112 on isometries, so they are among the finitely many X with Q(X) <= 112 (PARI qfminim, checked complete against the
        exact enumerator by produce/exact_qfminim.py).
        Ts から作る 32 個のブロック行列の Z-格子 H = { Σ y_k X_k が整数 } の中の、R の等長変換すべて（符号を除く）。H 上の
        Q(X) = tr(G_R^{-1} X G_R X^T) は正定値で等長変換では 112 なので、Q(X) ≤ 112 の有限個の中にある（PARI の qfminim。produce/exact_qfminim.py で厳密な列挙と照合して完全性を確かめる）。"""
        from sage.all import QQ, ZZ, lcm, matrix, pari
        gens = self.block_generators(Ts)
        k = len(gens)
        # H is dual to the Z-span of the coefficient rows (one row per matrix entry).
        # H は係数の行（行列の成分ごとに 1 行）の Z-スパンの双対。
        rows = {tuple(col) for col in zip(*[g.list() for g in gens])}
        rows.discard(tuple([QQ(0)] * k))
        rows = [list(r) for r in rows]
        d = lcm([QQ(x).denominator() for r in rows for x in r])
        span = matrix(ZZ, 0, k)
        for i in range(0, len(rows), 200):           # add rows in chunks to keep the HNF small / HNF を小さく保つため少しずつ足す
            span = span.stack(matrix(ZZ, [[int(d * x) for x in r] for r in rows[i:i + 200]])).hermite_form(include_zero_rows=False)
        S = span.change_ring(QQ) / d
        assert S.nrows() == k
        Hb = S.inverse().transpose()                 # rows: a basis of H / 行が H の基底
        Xs = []
        for row in Hb.rows():
            X = sum(c * g for c, g in zip(row, gens))
            assert all(x in ZZ for x in X.list())
            Xs.append(X.change_ring(ZZ))
        Ys = [self.GRi * X * self.GR for X in Xs]
        Q = matrix(ZZ, [Y.list() for Y in Ys]) * matrix(ZZ, [X.list() for X in Xs]).transpose()
        Q = Q + Q.transpose()                        # 2 Q, integral / 2Q（整数）
        assert Q.is_positive_definite()
        cands = qfminim_checked(Q, 2 * 112)
        isos = [X for X in (sum(ci * Xi for ci, Xi in zip(c, Xs)) for c in cands) if X * self.GR * X.transpose() == self.GR]
        return isos, len(cands), k

    def centralizer(self):
        """C_A(S): the 48 automorphisms of R commuting with SL_2(29) (both signs).
        C_A(S): SL_2(29) と可換な R の自己同型 48 個（符号込み）。"""
        half, _, _ = self.isometries_from(self.Mb)
        return half + [-X for X in half]

    def outer_automorphism(self):
        """The automorphism X of R inducing the outer automorphism of SL_2(29), from
        certificates/outer_automorphism_R.txt (checked: an isometry normalizing S, not in S C_A(S)
        is checked by the scripts that use it).
        SL_2(29) の外部自己同型を与える R の自己同型 X（certificates/outer_automorphism_R.txt）。等長変換で S を正規化することを確かめる。"""
        from sage.all import ZZ, matrix
        X = matrix(ZZ, certs.load_matrix("outer_automorphism_R.txt", 112, self.cert_dir).tolist())
        assert X * self.GR * X.transpose() == self.GR
        Xi = X.inverse()
        return X, Xi


def intertwiners(A, B, n):
    """Q-basis of { T in M_n(Q) : A_k T = T B_k for all k } (kernel modulo primes, Chinese remainder
    theorem and rational reconstruction; the result is checked exactly).
    { T ∈ M_n(Q) : すべての k で A_k T = T B_k } の Q-基底（素数を法とした核、中国剰余定理、有理数復元。結果は厳密に確かめる）。"""
    from sage.all import GF, QQ, ZZ, identity_matrix, matrix, previous_prime

    def kernel_mod(p):
        F = GF(p)
        I = identity_matrix(F, n)
        M = None
        for a, b in zip(A, B):
            K = a.change_ring(F).tensor_product(I) - I.tensor_product(b.change_ring(F).transpose())
            M = K if M is None else M.stack(K)
        return M.right_kernel().basis_matrix().echelon_form()
    p, P, acc, piv = ZZ(2 ** 23), ZZ(1), None, None
    for _ in range(60):
        p = previous_prime(p)
        K = kernel_mod(p).change_ring(ZZ)
        if K.nrows() == 0:
            return []
        if acc is None:
            acc, P, piv = K, p, K.pivots()
        else:
            if K.pivots() != piv:
                continue
            u = P * P.inverse_mod(p)
            acc = (acc + (K - acc) * u) % (P * p)
            P = P * p
        try:
            rows = [[QQ(x.rational_reconstruction(P)) if x else QQ(0) for x in r] for r in acc.rows()]
        except (ArithmeticError, ValueError):
            continue
        cand = [matrix(QQ, n, n, r) for r in rows]
        if all(a * T == T * b for T in cand for a, b in zip(A, B)):
            return cand
    raise RuntimeError("rational reconstruction did not converge")


def commutant_in_end_S(st, hs):
    """Basis (coefficients w.r.t. the 32 block generators) of the subspace of End_S(V) commuting with
    all matrices in hs (R-basis). / End_S(V) のうち hs のすべてと可換な部分空間の基底（32 個のブロック生成元に関する係数）。"""
    from sage.all import QQ, identity_matrix, matrix
    gens32 = st.block_generators(st.Mb)
    K = identity_matrix(QQ, 32)
    for h in hs:
        hq = h.change_ring(QQ)
        Ys = [sum(c * g for c, g in zip(row, gens32)) for row in K.rows()]
        K = matrix(QQ, [(Y * hq - hq * Y).list() for Y in Ys]).left_kernel().basis_matrix() * K
        if K.nrows() == 0:
            break
    return K, gens32
