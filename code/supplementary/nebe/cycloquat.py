r"""Nebe's lattices L_{2(p-1),2}(M) built from the definition [Neb98 §2, §3] (used for p = 29).
Requires SageMath.
Nebe の格子 L_{2(p-1),2}(M) を定義から作る [Neb98 §2, §3]（p = 29 で使う）。SageMath が要る。

Only primes p = 5 (mod 8) are handled. / p ≡ 5 (mod 8) の素数だけを扱う。

Model (a concrete form of [Neb98, Remark 3.3]; this explicit form is not written in the paper):
模型（[Neb98, Remark 3.3] を具体化したもの。この形そのものは論文には書かれていない）:
  * Q = Q(zeta_p) + Q(zeta_p) b with b^2 = -1 and b x = conj(x) b, a quaternion algebra over
    F+ = Q(zeta_p)^+ of Q-dimension 2(p-1).
    Q = Q(ζ_p) ⊕ Q(ζ_p) b（b^2 = -1、b x = conj(x) b）。F+ = Q(ζ_p)^+ 上の四元数環で Q-次元 2(p-1)。
  * Borel subgroup of SL_2(p): z = left multiplication by zeta; a(v) = tau~(v) b, where tau~ applies
    the generator of Gal(Q(zeta)/Q(sqrt p)) to both components; a^((p-1)/2) = -1.
    SL_2(p) の Borel 部分群: z = 左から ζ 倍、a(v) = tau~(v) b（tau~ は Gal(Q(ζ)/Q(√p)) の生成元を両成分に施す）。
  * The endomorphisms commuting with these are right multiplications by H = K4 + K4 b, where K4 is
    the quartic subfield of Q(zeta_p); H is the quaternion algebra over K = Q(sqrt p) ramified only
    at the two infinite places.
    これらと可換な自己準同型は H = K4 ⊕ K4 b の右からの積。K4 は Q(ζ_p) の 4 次部分体。H は K = Q(√p) 上の、2 つの無限素点だけで分岐する四元数環。
  * Invariant forms: Tr_{Q(zeta)/Q}(c (x conj(x') + y conj(y'))) with c in K totally positive.
    不変な内積: Tr_{Q(ζ)/Q}(c (x conj(x') + y conj(y')))、c ∈ K は総正。
  * For each maximal order M of H, exactly one of the lattices Z[zeta] M (1 - zeta)^i can be made
    unimodular; it is L_{2(p-1),2}(M) [Neb98, Thm 2.4, Prop 3.2].
    H の各極大整環 M について、Z[ζ] M (1-ζ)^i のうちユニモジュラーにできるものがちょうど 1 つあり、それが L_{2(p-1),2}(M)。
The resulting Gram matrix is checked directly to be integral, even and of determinant 1.
得られた Gram 行列が整数・偶・行列式 1 であることを最後に直接確かめる。

Adapted from the project's library (eeul/candidates/cycloquat.py) for the release: unused functions
removed, comments made bilingual; the computations are unchanged.
プロジェクトのライブラリ（eeul/candidates/cycloquat.py）を公開用に整えたもの。使わない関数を除き、注釈を英日併記にした。計算は変えていない。
"""
import itertools
import random

from sage.all import (GF, QQ, ZZ, CyclotomicField, QuadraticForm, block_matrix, identity_matrix, lcm, matrix,
                      primitive_root, vector, zero_matrix)


class CycloQuat:
    def __init__(self, p):
        assert p % 8 == 5, "only p = 5 (mod 8) / p ≡ 5 (mod 8) のみ"
        self.p = p
        self.F = F = CyclotomicField(p)
        self.z = z = F.gen()
        self.g = g = primitive_root(p)
        self.conj = F.hom([z ** (p - 1)])
        self.tau = F.hom([z ** pow(g, 2, p)])        # generator of Gal(Q(zeta)/Q(sqrt p)) / Gal(Q(ζ)/Q(√p)) の生成元
        q = (p - 1) // 4
        # Integral basis of K4: Gaussian periods. / K4 の整基底: ガウス周期。
        self.eta = [sum(z ** pow(g, 4 * k + j, p) for k in range(q)) for j in range(4)]
        self.sqrtp = self.eta[0] + self.eta[2] - self.eta[1] - self.eta[3]   # +-sqrt p
        assert self.sqrtp ** 2 == p
        # Q-basis of H: eta_j and eta_j b; structure constants. / H の Q-基底 eta_j, eta_j b と構造定数。
        self.hbasis = [(e, F(0)) for e in self.eta] + [(F(0), e) for e in self.eta]
        self.sc = [[self.hcoords(self.qmul(u, v)) for v in self.hbasis] for u in self.hbasis]
        self.one = self.hcoords((F(1), F(0)))
        # Trace form Tr_{H/Q}(u v). / トレース形式。
        self.Tmat = matrix(QQ, 8, 8, lambda i, j: self._tr4(self.qmul(self.hbasis[i], self.hbasis[j])[0]))
        # Bilinear form of Tr_{K/Q}(nrd(u)): B(u, v) = Tr_{K/Q}(trd(u v*)). / Tr_{K/Q}(nrd(u)) の双線形版。
        star = lambda u: (self.conj(u[0]), -u[1])   # noqa: E731
        self.Nmat = matrix(QQ, 8, 8, lambda i, j: self._tr4(self.qmul(self.hbasis[i], star(self.hbasis[j]))[0]))

    # --- arithmetic in Q and H / Q と H の演算 ---
    def qmul(self, u, v):
        c = self.conj
        return (u[0] * v[0] - u[1] * c(v[1]), u[0] * v[1] + u[1] * c(v[0]))

    def _normal(self, x):
        """Coefficients of x in Q(zeta) in the normal basis zeta^1..zeta^(p-1) (index k = 1..p-1).
        Q(ζ) の元の、正規基底 ζ^1..ζ^(p-1) での係数（添字 k = 1..p-1）。"""
        a = x.list()
        p = self.p
        return {k: (a[k] - a[0] if k < p - 1 else -a[0]) for k in range(1, p)}

    def _k4(self, x):
        n = self._normal(x)
        c = [n[pow(self.g, j, self.p)] for j in range(4)]
        assert sum(ci * e for ci, e in zip(c, self.eta)) == x, "not an element of K4 / K4 の元でない"
        return c

    def _tr4(self, x):
        """Tr_{K4/Q}(x) (Tr(eta_j) = -1). / K4 の元の Tr_{K4/Q}（Tr(eta_j) = -1）。"""
        return -sum(self._k4(x))

    def hcoords(self, u):
        return vector(QQ, self._k4(u[0]) + self._k4(u[1]))

    def helt(self, v):
        return (sum(v[j] * self.eta[j] for j in range(4)), sum(v[4 + j] * self.eta[j] for j in range(4)))

    def hmul(self, u, v):
        return sum(u[i] * v[j] * self.sc[i][j] for i in range(8) for j in range(8) if u[i] and v[j])

    def nrd(self, v):
        """Reduced norm of an element of H (coordinates), as (A, B) meaning A + B sqrt p in K.
        H の元（座標）の被約ノルム。K の元 A + B√p を (A, B) で返す。"""
        x, y = self.helt(v)
        n = x * self.conj(x) + y * self.conj(y)
        c = self._k4(n)        # element of K: c0 = c2, c1 = c3; 1 = -(sum eta), sqrt p = eta0 + eta2 - eta1 - eta3
        assert c[0] == c[2] and c[1] == c[3]
        return (-(c[0] + c[1]) / 2, (c[0] - c[1]) / 2)

    # --- lattices in H (8 x 8 rational matrices, rows = basis) / H の中の格子（8×8 の有理行列、行が基底）---
    @staticmethod
    def _lat(rows):
        """Basis matrix of the lattice spanned by rational vectors. / 有理ベクトルが張る格子の基底行列。"""
        M = matrix(QQ, rows)
        d = lcm([x.denominator() for x in M.list()])
        H = (d * M).change_ring(ZZ).hermite_form(include_zero_rows=False)
        return H.change_ring(QQ) / d

    def lmul_matrix(self, x):
        """Matrix of left multiplication by x (row vector v -> coordinates of x v).
        x を左から掛ける写像の行列（行ベクトル v -> x v の座標）。"""
        return matrix(QQ, [self.hmul(x, e) for e in identity_matrix(QQ, 8).rows()])

    def rmul_matrix(self, x):
        return matrix(QQ, [self.hmul(e, x) for e in identity_matrix(QQ, 8).rows()])

    def order_of(self, J, side):
        """Left order {x : x J in J} (side='left') or right order (side='right') of the lattice J.
        格子 J の左整環 {x : xJ ⊂ J}（side='left'）または右整環（'right'）。"""
        Ji = J.inverse()
        cols = []
        for k in range(8):
            e = identity_matrix(QQ, 8)[k]
            A = self.lmul_matrix(e) if side == "left" else self.rmul_matrix(e)
            cols.append((J * A * Ji).list())         # action of x = e_k in the basis of J / J の基底での x = e_k の作用
        W = matrix(QQ, cols)                         # 8 x 64; the order is {t : t W in Z^64} / 整環は {t : tW ∈ Z^64}
        d = lcm([x.denominator() for x in W.list()])
        Wz = (d * W).change_ring(ZZ)
        # t W in Z^64 iff t lies in the dual of the lattice spanned by the columns of W.
        # tW ∈ Z^64 ⟺ t は W の列が張る格子の双対に入る。
        C = Wz.transpose().hermite_form(include_zero_rows=False)
        assert C.nrows() == 8
        return self._lat((d * C.change_ring(QQ).inverse().transpose()).rows())

    def lat_mul(self, A, B):
        return self._lat([self.hmul(a, b) for a in A.rows() for b in B.rows()])

    def trace_det(self, L):
        return (L * self.Tmat * L.transpose()).det()

    def maximal_order(self):
        """One maximal order containing O_{K4} + O_{K4} b. / O_{K4} ⊕ O_{K4} b を含む極大整環を 1 つ。"""
        p = self.p
        L = identity_matrix(QQ, 8)
        target = p ** 4                              # d_K^4 (unramified at the finite places) / 有限素点で不分岐
        for _ in range(20):
            T = (L * self.Tmat * L.transpose()).change_ring(ZZ)
            if abs(T.det()) == target:
                return L
            ker = T.change_ring(GF(p)).kernel().basis_matrix()
            rad = self._lat(list(p * L) + [vector(QQ, [ZZ(x) for x in r]) * L for r in ker.rows()])
            L2 = self.order_of(rad, "left")
            if L2 == L:
                break
            L = L2
        # Hereditary but not maximal (an Eichler order of level p): go to a maximal order through an
        # idempotent of L/rad = F_p x F_p.
        # 遺伝的だが極大でない（レベル p の Eichler 整環）: L/rad ≅ F_p × F_p の冪等元から極大整環へ進む。
        T = (L * self.Tmat * L.transpose()).change_ring(ZZ)
        ker = T.change_ring(GF(p)).kernel().basis_matrix()
        rad = self._lat(list(p * L) + [vector(QQ, [ZZ(x) for x in r]) * L for r in ker.rows()])
        rnd = random.Random(p)
        Fp = GF(p)
        for _ in range(200):
            t = sum(rnd.randrange(p) * r for r in L.rows())
            # Left action of t on L/rad (2-dimensional). / t の L/rad（2 次元）への左作用。
            A = matrix(QQ, [self.hmul(t, r) for r in L.rows()]) * L.inverse()
            A = A.change_ring(ZZ).change_ring(Fp)
            R = (rad * L.inverse()).change_ring(ZZ).change_ring(Fp).row_space()
            V = Fp ** 8
            W, bas = R, []
            for v in V.basis():
                if v not in W:
                    bas.append(v)
                    W = W + V.span([v])
            if len(bas) != 2:
                raise ValueError(f"dim L/rad = {len(bas)}")
            full = matrix(Fp, list(R.basis()) + bas)
            Aq = (matrix(Fp, bas) * A * full.inverse()).matrix_from_columns(range(6, 8))
            rts = Aq.charpoly().roots(multiplicities=False)
            if len(rts) != 2:
                continue
            r1, r2 = rts
            e = (t - ZZ(r2) * self.one) * ZZ((r1 - r2) ** (-1))
            I = self._lat(list(rad) + [self.hmul(r, e) for r in L.rows()])
            M = self.order_of(I, "right")
            if abs(self.trace_det(M)) == target:
                return M
            M = self.order_of(I, "left")
            if abs(self.trace_det(M)) == target:
                return M
        raise ValueError("no maximal order found / 極大整環が見つからない")

    def norm_gram(self, L):
        """Gram matrix of the Z-lattice (L, Tr_{K/Q} nrd) (Q(u) = Tr nrd(u) = (1/2) u G u^T).
        Z-格子 (L, Tr_{K/Q} nrd) の Gram 行列。"""
        return L * self.Nmat * L.transpose()

    def units_mod_center(self, M):
        """[M^x : R^x] = #{u in M : nrd(u) = 1} / 2. / [M^× : R^×] = #{u ∈ M : nrd(u) = 1} / 2。"""
        G = self.norm_gram(M).change_ring(ZZ)
        vs = QuadraticForm(ZZ, G).short_vector_list_up_to_length(3)[2]      # Q = (1/2) v G v^T = Tr nrd = 2
        return sum(1 for v in vs if self.nrd(vector(QQ, v) * M) == (1, 0)) // 2

    # --- the lattice of dimension 2(p-1) / 2(p-1) 次元の格子 ---
    def lattice(self, M, choice=0):
        r"""(Gram matrix, basis matrix) of L_{2(p-1),2}(M). The basis consists of row vectors in
        Q(zeta)^2 = Q^{2(p-1)} (coefficients of powers of zeta).
        L_{2(p-1),2}(M) の (Gram 行列, 基底行列)。基底は Q(ζ)^2 = Q^{2(p-1)}（ζ 冪の係数）の行ベクトル。"""
        from sage.all import QuadraticField, factor
        p, z, n = self.p, self.z, self.p - 1
        gens = [self.helt(r) for r in M.rows()]
        gens = gens + [(-self.conj(y), self.conj(x)) for (x, y) in gens]       # also b M (for stability under a) / b M も入れる
        Mz = matrix(QQ, [(z ** (k + 1)).list() for k in range(n)])             # matrix of x -> zeta x / x -> ζx の行列
        B0 = matrix(QQ, n, n, lambda i, j: (z ** i * self.conj(z ** j)).trace())   # Tr(zeta^i conj zeta^j)
        rows = []
        for (x, y) in gens:
            vx, vy = vector(QQ, x.list()), vector(QQ, y.list())
            for k in range(n):
                rows.append(list(vx) + list(vy))
                vx, vy = vx * Mz, vy * Mz
        d = lcm([QQ(x).denominator() for r in rows for x in r])
        Lb = (d * matrix(QQ, rows)).change_ring(ZZ).hermite_form(include_zero_rows=False).change_ring(QQ) / d
        assert Lb.nrows() == 2 * n
        one_minus = identity_matrix(QQ, n) - Mz
        Z = zero_matrix(QQ, n, n)
        Sh = block_matrix(QQ, [[one_minus, Z], [Z, one_minus]], subdivide=False)
        BB = block_matrix(QQ, [[B0, Z], [Z, B0]], subdivide=False)
        # Inner product Tr(c (x conj x' + y conj y')), c in K totally positive. The determinant fixes
        # (1 - zeta)^i and c: (1 - zeta)^i multiplies det by p^{4i}, c by N_{K/Q}(c)^{p-1}.
        # 内積 Tr(c(x x̄' + y ȳ'))、c ∈ K は総正。行列式から (1-ζ)^i と c を決める: (1-ζ)^i で det は p^{4i} 倍、c で N_{K/Q}(c)^{p-1} 倍。
        det0 = (Lb * BB * Lb.transpose()).det()
        Kq = QuadraticField(p, "s")
        s_ = Kq.gen()
        eps = Kq.units()[0]
        assert eps.norm() == -1

        def tp(x):                                     # totally positive associate / 総正な同伴元
            if x.norm() < 0:
                x = x * eps
            return x if x.trace() > 0 else -x
        choices, v = [], 0
        for q, e in factor(det0):
            if q == p:
                v = e
                continue
            assert e % n == 0, f"the power {e} of {q} in det is not divisible by p - 1"
            f = -e // n                                # power of q in N(c) / N(c) の q 冪
            P = Kq.ideal(q).factor()
            if len(P) == 2:
                g1, g2 = [tp(I.gens_reduced()[0]) for I, _ in P]
                choices.append([g1 ** a_ * g2 ** (f - a_) for a_ in range(min(0, f) - 1, max(0, f) + 2)])
            else:
                assert f % 2 == 0
                choices.append([Kq(q) ** (f // 2)])
        pi = tp(s_)
        sol = [(i, k) for k in range(-3, 4) for i in range(0, (p - 1) // 2) if v + 4 * i - (p - 1) * k == 0]
        assert sol, f"cannot remove the power {v} of p from det"
        # There are two solutions (i and i + (p-1)/4; isometric by [Neb98, Prop 3.2 (ii)], but only one is
        # invariant under SL_2(p)); `choice` selects one.
        # 解は 2 つある（i と i + (p-1)/4。互いに等長 [Neb98 Prop 3.2 (ii)] だが、SL_2(p) で不変なのは片方だけ）。choice で選ぶ。
        self.n_choices = len(sol)
        i, k = sol[choice]
        Lb2 = Lb * Sh ** i
        ok = None
        for combo in itertools.product(*choices):
            c = pi ** (-k)
            for t in combo:
                c = c * t
            a_, b_ = c.list()
            cm = self._mulmat(a_ + b_ * self.sqrtp)
            C2 = block_matrix(QQ, [[cm, Z], [Z, cm]], subdivide=False)
            G = Lb2 * C2 * BB * Lb2.transpose()
            if all(x in ZZ for x in G.list()) and G.det() == 1:
                ok = (G, C2, c)
                break
        assert ok is not None, "no c makes the lattice unimodular / ユニモジュラーにする c が見つからない"
        G, C2, c = ok
        G = G.change_ring(ZZ)
        assert G.is_symmetric() and all(G[j, j] % 2 == 0 for j in range(2 * n)), "not even / 偶でない"
        self.last = dict(i=i, k=k, c=c, basis=Lb2, form=C2 * BB)
        return G, Lb2

    def _mulmat(self, x):
        n, z = self.p - 1, self.z
        return matrix(QQ, [(z ** k * x).list() for k in range(n)])


def _k_elt(X, a, b):
    """The element a + b sqrt p of K, in coordinates of H. / K の元 a + b√p を H の座標で。"""
    return a * X.one + b * X.hcoords((X.sqrtp, X.F(0)))


def is_principal_with_norm(X, I, nA, nB):
    """An element x of the lattice I with nrd(x) = nA + nB sqrt p, or None (by enumerating all vectors
    with Tr nrd = 2 nA). / 格子 I の中で nrd(x) = nA + nB√p となる元（無ければ None）。Tr nrd = 2nA のベクトルを全列挙する。"""
    G = X.norm_gram(I)
    d = lcm([x.denominator() for x in G.list()])
    Gz = (d * G).change_ring(ZZ)
    target = 2 * nA                                    # Tr_{K/Q}(nA + nB sqrt p)
    if (d * target) not in ZZ:
        return None
    vs = QuadraticForm(ZZ, Gz).short_vector_list_up_to_length(ZZ(d * target) + 1)[ZZ(d * target)]
    for v in vs:
        x = vector(QQ, v) * I
        if X.nrd(x) == (nA, nB):
            return x
    return None


def maximal_order_classes(X, verbose=False, max_steps=60):
    """Representatives of the conjugacy classes of maximal orders of H (conjugation over K), as
    [(M, [M^x : R^x])]. Completeness is checked with the mass formula.
    Method: walk through q-neighbours (right orders of left ideals I = M x + q M) breadth first; two
    orders are in the same class iff the connecting ideal is principal, decided exactly by enumerating
    short vectors.
    H の極大整環の共役類の代表（K 上の共役で分類）。返り値 [(M, [M^×:R^×])]。完全性は質量公式で確かめる。
    方法: q-隣接（左イデアル I = Mx + qM の右整環）をたどる。同じ類かどうかは、つなぐイデアルが単項かどうかを短いベクトルの全列挙で厳密に判定する。"""
    from sage.all import QuadraticField, kronecker_symbol, next_prime, sigma
    p = X.p
    Kq = QuadraticField(p, "s")
    s = Kq.gen()
    # zeta_K(-1) = (1/60) sum_{b odd, b^2 < p} sigma_1((p - b^2)/4); the mass is zeta_K(-1)/2.
    # ζ_K(-1) = (1/60) Σ_{b 奇, b^2 < p} σ_1((p - b^2)/4)。質量は ζ_K(-1)/2。
    zeta = QQ(sum(sigma((p - b * b) // 4, 1) for b in range(-int(p ** 0.5) - 1, int(p ** 0.5) + 2) if b % 2 and b * b < p)) / 60
    mass = zeta / 2
    assert Kq.class_number() == 1
    eps = Kq.units()[0]
    M0 = X.maximal_order()
    # each class: (M, w, I, nI); I a left-M0, right-M ideal, nI a totally positive generator of its reduced norm
    # 各類: (M, w, I, nI)。I は左 M0・右 M のイデアル、nI はその被約ノルムの総正な生成元。
    classes = [(M0, X.units_mod_center(M0), M0, Kq(1))]
    total = QQ(1) / classes[0][1]
    if verbose:
        print(f"  mass {mass}; class 0: w = {classes[0][1]}", flush=True)
    rnd = random.Random(p)
    q, steps, primes = 2, 0, []
    while len(primes) < 3:
        q = next_prime(q)
        if q != p and kronecker_symbol(p, q) == 1:
            primes.append(q)
    while total < mass and steps < max_steps:
        steps += 1
        ci = rnd.randrange(len(classes))
        M, w, I, nI = classes[ci]
        qq = primes[rnd.randrange(len(primes))]
        r = [t for t in range(qq) if (t * t - p) % qq == 0][rnd.randrange(2)]
        # x in M with nrd(x) in q = (q, sqrt p - r) and x not in q M. / x ∈ M、nrd(x) ∈ q、x ∉ qM。
        for _ in range(2000):
            x = sum(rnd.randrange(-2, 3) * row for row in M.rows())
            A, B = X.nrd(x)
            val = A + B * r
            if x != 0 and val.denominator() % qq and (val.numerator() % qq == 0) and not all((c / qq) in ZZ for c in M.solve_left(x)):
                break
        else:
            continue
        gens = [X.hmul(m, x) for m in M.rows()] + [qq * m for m in M.rows()] + [X.hmul(m, _k_elt(X, -r, 1)) for m in M.rows()]
        J = X._lat(gens)                               # left M-ideal of norm q / ノルム q の左 M-イデアル
        Mn = X.order_of(J, "right")
        assert abs(X.trace_det(Mn)) == p ** 4
        gq = Kq.ideal(qq, s - r).gens_reduced()[0]     # totally positive generator of q / q の総正な生成元
        if gq.norm() < 0:
            gq = gq * eps
        if gq.trace() < 0:
            gq = -gq
        In = X.lat_mul(I, J)
        nIn = nI * gq
        # Compare with the known classes: is conj(I_j) I_n principal (norm nI_j nI_n, up to squares of units)?
        # 既知の類と比べる: conj(I_j)·I_n は単項か（ノルム nI_j·nI_n、単数の平方倍を許す）。
        star = lambda v: X.hcoords((X.conj(X.helt(v)[0]), -X.helt(v)[1]))   # noqa: E731
        found = False
        for (Mj, wj, Ij, nIj) in classes:
            C = X.lat_mul(X._lat([star(v) for v in Ij.rows()]), In)
            t = nIj * nIn
            best = min([t * eps ** (2 * e) for e in range(-6, 7)], key=lambda u: u.trace())
            a_, b_ = best.list()
            if is_principal_with_norm(X, C, a_, b_) is not None:
                found = True
                break
        if not found:
            wn = X.units_mod_center(Mn)
            classes.append((Mn, wn, In, nIn))
            total += QQ(1) / wn
            if verbose:
                print(f"  class {len(classes) - 1}: w = {wn} (total {total} / {mass})", flush=True)
    assert total == mass, f"mass formula not satisfied: {total} != {mass}"
    return [(M, w) for (M, w, _, _) in classes]


def weyl_element(X, signs=(1, 1), prec=600):
    r"""Action of the Weyl element w = [[0, 1], [-1, 0]] of SL_2(p) (a rational matrix acting on the
    right of row vectors of Q(zeta)^2 = Q^{2(p-1)}).
    SL_2(p) の Weyl 元 w = [[0, 1], [-1, 0]] の作用（Q(ζ)^2 = Q^{2(p-1)} の行ベクトルに右から掛ける有理行列）。

    In the embedding coordinates v_k = (sigma_k(x), sigma_k(y)), z multiplies by zeta^k and
    (a v)_k = J v_{k g^2} (J = (x, y) -> (-y, x)). This is isomorphic, as a representation of the Borel
    subgroup, to the sum of two copies each of the two odd parts of the Weil representation
    (psi(c x^2), c = 1, g), identified by v_{c g^{2j}} = J^j F_{g^j}; the Weyl element is the Fourier
    transform (w F)_x = kappa_c sum_y psi(2 c x y) F_y, kappa_c = +-1/sqrt p. Endomorphisms commuting with
    the Borel subgroup commute with SL_2(p) [Neb98, proof of Prop. 3.2], so the identification does not matter.
    埋め込み座標 v_k = (σ_k(x), σ_k(y)) では z は ζ^k 倍、(av)_k = J v_{kg^2}（J = (x,y) -> (-y,x)）。これは Borel 部分群の表現として、
    Weil 表現の奇関数部分 2 種（ψ(cx^2)、c = 1, g）の 2 個ずつの直和と同型で、v_{cg^{2j}} = J^j F_{g^j} と同一視できる。Weyl 元は
    Fourier 変換 (wF)_x = κ_c Σ_y ψ(2cxy) F_y、κ_c = ±1/√p。Borel 部分群と可換な自己準同型は SL_2(p) とも可換 [Neb98 Prop 3.2 の証明]
    なので、同一視の取り方に依らない。

    Computed numerically with `prec` bits and rounded to rationals. **The caller must check correctness
    exactly** (an integral matrix preserving the lattice and its Gram matrix). signs = signs of (kappa_1, kappa_g).
    prec ビットの数値計算で求めて有理数に丸める。**正しさは呼び出し側で厳密に確かめること**（格子を保つ整数行列で、Gram 行列を保つ）。
    """
    from sage.all import ComplexField, I as II, RealField, exp, pi as PI, sqrt
    p, n, g = X.p, X.p - 1, X.g
    C, R = ComplexField(prec), RealField(prec)
    zeta = exp(2 * C(PI) * C(II) / p)
    Phi = matrix(C, n, n, lambda k, i: zeta ** (((k + 1) * i) % p))       # row k <-> embedding sigma_{k+1}
    Phii = Phi.inverse()
    # w in embedding coordinates, acting on column vectors; order (x_1..x_{p-1}, y_1..y_{p-1}).
    # 埋め込み座標での w（列ベクトルに作用）。成分の並び (x_1..x_{p-1}, y_1..y_{p-1})。
    W = matrix(C, 2 * n, 2 * n)
    Jp = [matrix(ZZ, [[1, 0], [0, 1]]), matrix(ZZ, [[0, -1], [1, 0]]), matrix(ZZ, [[-1, 0], [0, -1]]), matrix(ZZ, [[0, 1], [-1, 0]])]
    sq = sqrt(R(p))
    for c, sg in zip((1, g), signs):
        kap = sg / sq
        for j in range(n // 2):
            kk = (c * pow(g, 2 * j, p)) % p            # output index / 出力の添字
            for i in range(n // 2):
                k = (c * pow(g, 2 * i, p)) % p         # input index / 入力の添字
                u = (2 * c * pow(g, i + j, p)) % p
                coef = kap * (zeta ** u - zeta ** (-u))
                Jm = Jp[(j - i) % 4]
                for a_ in range(2):
                    for b_ in range(2):
                        if Jm[a_, b_]:
                            W[a_ * n + kk - 1, b_ * n + k - 1] += coef * Jm[a_, b_]
    Z = zero_matrix(C, n, n)
    PP = block_matrix(C, [[Phi, Z], [Z, Phi]], subdivide=False)
    PPi = block_matrix(C, [[Phii, Z], [Z, Phii]], subdivide=False)
    Wq = PPi * W * PP
    err = max(abs(x.imag()) for x in Wq.list())
    den = p ** 2
    Wr = matrix(QQ, 2 * n, 2 * n, lambda i, j: QQ((Wq[i, j].real() * den).round()) / den)
    err2 = max(abs(Wq[i, j].real() - Wr[i, j]) for i in range(2 * n) for j in range(2 * n))
    return Wr.transpose(), (err, err2)                 # acts on the right of row vectors / 行ベクトルに右から
