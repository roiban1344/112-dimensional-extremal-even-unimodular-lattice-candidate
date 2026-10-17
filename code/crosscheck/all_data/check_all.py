#!/usr/bin/env python3
"""Independent verifier for the certificates of the 112-dimensional lattice R.

Release note: for the public repository only the input functions (section "inputs" below: the
four files of data/ and the certificate files of certificates/) and the output directory
(out/crosscheck_all_data/) were changed. The checks themselves are unchanged. It shares no code with
code/r112 or code/verify.
公開用の注記: 公開リポジトリ向けに変えたのは入力の関数（下の「inputs」の節。data/ の 4 ファイルと
certificates/ の検証用データを読む）と出力先（out/crosscheck_all_data/）だけ。検査そのものは変えていない。
code/r112・code/verify とはコードを共有しない。

Written from a specification of the certificates, WITHOUT reading or importing the code that
produced them.  Run with `python -I`.

Usage (from the repository root):
  python3 -I code/crosscheck/all_data/check_all.py <command> [options]
commands:
  v0        premises (Gram, generators, tau, group order, units)
  theta     extremal theta series of rank 56 (counts of norm 6 and norm 8 vectors)
  v1        minimal vectors: 656 orbits, canonical forms, sizes; builds the table M/{+-1}
  v2        odd-class certificates (656), all conditions
  v3cheap   even-class certificates (1,527,984): conditions 11, 12, 14 and the unit matching
  v3sample  condition 13 on 10,000 random even-class certificates (+ int64 cross-check)
  v3full    condition 13 on all even-class certificates (resumable, --part p or all)
  v4        norm-8 table: canonical forms of the 1,527,984 vectors x*u, orbit sizes (resumable)
  neg       negative controls
  summary   collect the results of all steps into summary.json

Conventions: vectors are integer row vectors, maps act on the right, (x,y) = x G y^T.

Exactness: every product computed in floating point (float32 BLAS) has all partial sums bounded,
from the actual inputs, by sum_i |a_i| |b_i| < 2^24, asserted in the code (in `InnerMax`, `InnerMax.ips` and `Orbits._canon`).
Every int64 product is guarded by a bound < 2^62 (`imul`).
"""
import sys, os

# BLAS threads: we parallelise with Python threads and single-threaded BLAS calls.
if "--threads" in sys.argv:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

import json, time, pickle, hashlib, platform, argparse, threading
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = os.path.join(ROOT, "data")            # the four published input files
C = os.path.join(ROOT, "certificates")    # the certificate files (certificates/MANIFEST.md)
OUT = os.path.join(ROOT, "out", "crosscheck_all_data")
CACHE = os.path.join(OUT, "cache")
N = 56
GROUP_ORDER = 24360
NPAIRS = 1527984
F32_LIMIT = 2 ** 24

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:9.1f}s]", *a, flush=True)


def atomic_write_bytes(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, path)


def atomic_json(path, obj):
    atomic_write_bytes(path, (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode())


def atomic_npz(path, **arrs):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrs)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.rename(tmp, path)


def atomic_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.rename(tmp, path)


def record(step, result):
    """Write the result of one step to OUT/<step>.json with machine information."""
    os.makedirs(OUT, exist_ok=True)
    result = dict(result)
    result["_step"] = step
    result["_machine"] = {"host": platform.node(), "platform": platform.platform(),
                          "python": platform.python_version(), "numpy": np.__version__,
                          "cpus": os.cpu_count(),
                          "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS")}
    result["_wall_seconds"] = round(time.time() - T0, 1)
    result["_finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    atomic_json(os.path.join(OUT, step + ".json"), result)
    log("wrote", os.path.join(OUT, step + ".json"))


def verdict(ok):
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- exact integer helpers

def imul(A, B):
    """Exact int64 product, with an overflow guard computed from the inputs."""
    A = np.asarray(A, dtype=np.int64)
    B = np.asarray(B, dtype=np.int64)
    ma = int(np.abs(A).max()) if A.size else 0
    mb = int(np.abs(B).max()) if B.size else 0
    assert ma * mb * A.shape[-1] < 2 ** 62, "int64 overflow risk"
    return A @ B


def assert_f32_exact(A, B):
    """A (n x k) @ B (k x m) in float32 is exact if all |partial sums| < 2^24.
    Every partial sum of row a and column b is bounded by sum_i |a_i| |b_i|
    <= (max_row sum_i |a_i|) * max |b|. Returns the bound."""
    bound = int(np.abs(np.asarray(A, dtype=np.int64)).sum(axis=1).max()) * int(np.abs(np.asarray(B, dtype=np.int64)).max())
    assert bound < F32_LIMIT, f"float32 not exact: bound {bound} >= 2^24"
    return bound


def bareiss_minors(M):
    """Leading principal minors of an integer matrix (fraction-free, no pivoting)."""
    A = [list(map(int, r)) for r in M]
    n = len(A)
    minors = []
    prev = 1
    for k in range(n):
        if A[k][k] == 0:
            minors.append(0)
            return minors
        minors.append(A[k][k])
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return minors


def exact_inverse(M):
    n = len(M)
    A = [[Fraction(int(x)) for x in r] + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    inv = [[A[i][n + j] for j in range(n)] for i in range(n)]
    assert all(x.denominator == 1 for r in inv for x in r), "inverse is not integral"
    return np.array([[int(x) for x in r] for r in inv], dtype=np.int64)


# ---------------------------------------------------------------- inputs

def read_matrices(path):
    mats, cur = [], []
    for line in open(path):
        s = line.strip()
        if s.startswith("#"):
            continue
        if not s:
            if cur:
                mats.append(cur)
                cur = []
            continue
        cur.append([int(t) for t in s.split()])
    if cur:
        mats.append(cur)
    for m in mats:
        assert len(m) == N and all(len(r) == N for r in m), path
    return [np.array(m, dtype=np.int64) for m in mats]


_cache = {}


def gram():
    if "G" not in _cache:
        (_cache["G"],) = read_matrices(os.path.join(D, "L56_gram.txt"))
    return _cache["G"]


def generators():
    g = read_matrices(os.path.join(D, "sl2_29_generators.txt"))
    assert len(g) == 2
    return g


def tau():
    (t,) = read_matrices(os.path.join(D, "tau.txt"))
    return t


def _npz(name, key):
    with np.load(os.path.join(C, name), allow_pickle=False) as z:
        return z[key]


def min_reps():
    return _npz("min_reps.npz", "reps").astype(np.int64)


def norm8_table():
    return _npz("norm8_reps.npz", "reps").astype(np.int64), _npz("norm8_reps.npz", "sizes").astype(np.int64)


def m_basis():
    B = [b.astype(np.int64) for b in _npz("commutant.npz", "basis")]
    assert len(B) == 8
    return B


def odd_certs():
    # Row j of odd_bounding.npz belongs to representative j, direction tau ("which" = 0).
    w0 = _npz("odd_bounding.npz", "w0").astype(np.int64)
    return np.arange(len(w0), dtype=np.int64), w0, np.zeros(len(w0), dtype=np.int64)


def even_cert_part(p):
    # The producer ran 6 interleaved parts; part p consisted of the pairs p, p + 6, p + 12, ...,
    # which are the rows p::6 of even_bounding.npz.
    return _npz("even_bounding.npz", "w0")[p::6].astype(np.int64)


def even_cert_json(p):
    # The release has no producer progress log. "done" = number of rows of the part; "unresolved"
    # is empty because every row is a candidate certificate, which conditions 11-13 then test.
    return {"done": len(even_cert_part(p)), "unresolved": []}


def norms(X, block=200000):
    G = gram()
    out = []
    for a in range(0, len(X), block):
        Xb = np.asarray(X[a:a + block], dtype=np.int64)
        out.append(np.einsum("ij,ij->i", imul(Xb, G), Xb))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------- group and units

def build_group():
    """All elements of <g1, g2> by breadth-first search (exact int64)."""
    path = os.path.join(CACHE, "group.npy")
    if os.path.exists(path):
        return np.load(path)
    gens = generators()
    I = np.eye(N, dtype=np.int64)
    seen = {I.astype(np.int16).tobytes(): 0}
    elems = [I]
    frontier = [I]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                b = imul(a, g)
                assert np.abs(b).max() < 2 ** 15
                k = b.astype(np.int16).tobytes()
                if k not in seen:
                    seen[k] = len(elems)
                    elems.append(b)
                    nxt.append(b)
        frontier = nxt
        assert len(elems) <= 10 ** 6
    arr = np.array(elems, dtype=np.int16)
    os.makedirs(CACHE, exist_ok=True)
    atomic_npy(path, arr)
    return arr


def find_units():
    """Isometries in the order M = Z<B_1..B_8>: enumerate c in Z^8 with
    Q(c) = tr(u G u^T G^{-1}) <= 56 (u = sum c_a B_a; Q is positive definite and Q = 56 on
    isometries) with PARI's qfminim on an integral form, then keep u with u G u^T = G."""
    path = os.path.join(CACHE, "units.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["units"], z["coeffs"], int(z["nshort"])
    import cypari2
    pari = cypari2.Pari()
    G = gram()
    Gi = exact_inverse(G)
    B = m_basis()
    Q = np.zeros((8, 8), dtype=object)
    for a in range(8):
        X = imul(B[a], G)
        for b in range(8):
            # tr(B_a G B_b^T G^-1) = sum_ij (B_a G)_ij (G^-1 B_b)_ij   (G^-1 symmetric)
            Q[a, b] = int(np.einsum("ij,ij->", X, imul(Gi, B[b])))
    assert all(Q[a, b] == Q[b, a] for a in range(8) for b in range(8))
    Qp = pari.matrix(8, 8, [int(Q[a, b]) for a in range(8) for b in range(8)])
    res = pari.qfminim(Qp, 56)
    vecs = res[2]
    nshort = int(res[0])  # PARI counts v and -v; the matrix holds one of each pair
    ncols = int(pari.matsize(vecs)[1])
    assert nshort == 2 * ncols
    units, coeffs = [], []
    for j in range(ncols):
        c = [int(vecs[i, j]) for i in range(8)]
        for s in (1, -1):
            cc = [s * x for x in c]
            u = sum(cc[a] * B[a] for a in range(8))
            if np.array_equal(imul(imul(u, G), u.T), G):
                units.append(u)
                coeffs.append(cc)
    units = np.array(units, dtype=np.int64)
    coeffs = np.array(coeffs, dtype=np.int64)
    os.makedirs(CACHE, exist_ok=True)
    atomic_npz(path, units=units, coeffs=coeffs, nshort=np.int64(nshort))
    return units, coeffs, nshort


def units_mod_sign():
    units, coeffs, _ = find_units()
    reps = []
    for u, c in zip(units, coeffs):
        # representative of {u, -u}: first nonzero coefficient positive
        nz = c[np.nonzero(c)[0][0]]
        if nz > 0:
            reps.append((tuple(c), u))
    reps.sort(key=lambda t: t[0])
    return np.array([u for _, u in reps]), np.array([c for c, _ in reps])


# ---------------------------------------------------------------- orbits and canonical forms

class Orbits:
    """Orbit computations under the group H = <g1, g2> (all GROUP_ORDER elements)."""

    def __init__(self, ncols=4):
        self.H = build_group().astype(np.int32)  # (|H|, 56, 56)
        assert self.H.shape[0] == GROUP_ORDER
        self.maxH = int(np.abs(self.H).max())
        self.k = ncols
        # first k columns of every element: C[i, g*k + j] = H[g, i, j]
        self.Ck = np.ascontiguousarray(self.H[:, :, :ncols].transpose(1, 0, 2).reshape(N, -1)).astype(np.float32)

    def images_full(self, x):
        """All images x h (h in H) of one vector, exact int64, shape (|H|, 56)."""
        x = np.asarray(x, dtype=np.int64)
        assert int(np.abs(x).sum()) * self.maxH < 2 ** 31
        return np.einsum("i,gij->gj", x.astype(np.int32), self.H).astype(np.int64)

    def canon(self, X, sub=256):
        X = np.asarray(X, dtype=np.int64)
        cs, ss = [], []
        for a in range(0, len(X), sub):
            c, s = self._canon(X[a:a + sub])
            cs.append(c)
            ss.append(s)
        return np.concatenate(cs), np.concatenate(ss)

    def _canon(self, X):
        """For each row x of X: lexicographically smallest vector of the orbit xH, and the
        orbit size |H| / |Stab(x)|. Computes the first k coordinates of all images in float32
        (exactness asserted) and resolves remaining ties with exact full images."""
        X = np.asarray(X, dtype=np.int64)
        # partial sums of x h are bounded by |x|_1 max|h|
        assert int(np.abs(X).sum(axis=1).max()) * self.maxH < F32_LIMIT
        X32 = X.astype(np.int32)
        P = (X.astype(np.float32) @ self.Ck).reshape(X.shape[0], GROUP_ORDER, self.k)
        P = np.rint(P).astype(np.int32)
        canon = np.empty_like(X)
        size = np.empty(X.shape[0], dtype=np.int64)
        for b in range(X.shape[0]):
            Pb = P[b]
            # lexicographic minimum on the first k coordinates
            cand = np.arange(GROUP_ORDER)
            for j in range(self.k):
                col = Pb[cand, j]
                cand = cand[col == col.min()]
                if len(cand) == 1:
                    break
            if len(cand) == 1:
                canon[b] = X32[b] @ self.H[cand[0]]
            else:
                full = np.einsum("i,gij->gj", X32[b], self.H[cand])
                order = np.lexsort(full.T[::-1])
                canon[b] = full[order[0]]
            # stabiliser: images equal to x
            same = np.nonzero((Pb == X[b, : self.k]).all(axis=1))[0]
            full = np.einsum("i,gij->gj", X32[b], self.H[same])
            stab = int((full == X[b]).all(axis=1).sum())
            assert stab >= 1 and GROUP_ORDER % stab == 0
            size[b] = GROUP_ORDER // stab
        return canon, size


# ---------------------------------------------------------------- the table of minimal vectors

def min_half():
    """M / {+-1}: every minimal vector up to sign, from the 656 orbit representatives
    (rows with first nonzero coordinate positive). Built in v1."""
    path = os.path.join(CACHE, "min-half.npy")
    return np.load(path, mmap_mode="r")


class InnerMax:
    """max_{u in M} |(w, u)| for many w, using float32 BLAS with an exactness assertion."""

    def __init__(self, threads=1, bw=512, bm=65536):
        Mh = np.asarray(min_half())
        self.maxM = int(np.abs(Mh).max())
        self.Mf = Mh.astype(np.float32)  # (7,795,200, 56)
        self.nM = self.Mf.shape[0]
        self.threads = threads
        self.bw, self.bm = bw, bm

    def _chunk(self, Af):
        buf = np.empty((Af.shape[0], self.bm), dtype=np.float32)
        mx = np.full(Af.shape[0], -np.inf, dtype=np.float32)
        mn = np.full(Af.shape[0], np.inf, dtype=np.float32)
        for a in range(0, self.nM, self.bm):
            Mb = self.Mf[a:a + self.bm]
            out = buf[:, : Mb.shape[0]]
            np.matmul(Af, Mb.T, out=out)
            np.maximum(mx, out.max(axis=1), out=mx)
            np.minimum(mn, out.min(axis=1), out=mn)
        return np.maximum(mx, -mn)

    def ips(self, w):
        """Exact inner products (w, u) for all u in M/{+-1} (int64 vector)."""
        a = imul(np.asarray(w, dtype=np.int64)[None], gram())[0]
        assert int(np.abs(a).sum()) * self.maxM < F32_LIMIT
        r = self.Mf @ a.astype(np.float32)
        ri = np.rint(r).astype(np.int64)
        assert np.array_equal(ri.astype(np.float32), r)
        return ri

    def __call__(self, W):
        W = np.asarray(W, dtype=np.int64)
        A = imul(W, gram())  # (w, u) = (w G) . u
        bound = int(np.abs(A).sum(axis=1).max()) * self.maxM
        assert bound < F32_LIMIT, f"float32 not exact: bound {bound}"
        Af = A.astype(np.float32)
        chunks = [Af[i:i + self.bw] for i in range(0, len(Af), self.bw)]
        if self.threads > 1:
            with ThreadPoolExecutor(self.threads) as ex:
                res = list(ex.map(self._chunk, chunks))
        else:
            res = [self._chunk(c) for c in chunks]
        r = np.concatenate(res) if res else np.zeros(0, dtype=np.float32)
        ri = np.rint(r).astype(np.int64)
        assert np.array_equal(ri.astype(np.float32), r)
        return ri, bound


def inner_max_int64(W, block=32768):
    """The same quantity by plain int64 numpy (no BLAS), for cross-checks."""
    W = np.asarray(W, dtype=np.int64)
    A = imul(W, gram())
    Mh = min_half()
    best = np.zeros(len(W), dtype=np.int64)
    for a in range(0, Mh.shape[0], block):
        Mb = np.asarray(Mh[a:a + block], dtype=np.int64)
        P = imul(A, Mb.T)
        best = np.maximum(best, np.abs(P).max(axis=1))
    return best


# ---------------------------------------------------------------- steps

def step_theta(args):
    """Extremal theta series of an even unimodular lattice of rank 56: the unique
    f = sum a_n q^n in C[E4^7, E4^4 Delta, E4 Delta^2] with a_0 = 1, a_1 = a_2 = 0."""
    P = 8

    def sigma3(n):
        return sum(d ** 3 for d in range(1, n + 1) if n % d == 0)

    def mul(a, b):
        c = [0] * P
        for i, x in enumerate(a):
            if x:
                for j in range(P - i):
                    c[i + j] += x * b[j]
        return c

    def pw(a, e):
        r = [1] + [0] * (P - 1)
        for _ in range(e):
            r = mul(r, a)
        return r

    E4 = [1] + [240 * sigma3(n) for n in range(1, P)]
    # Delta = q prod (1 - q^n)^24
    prod = [1] + [0] * (P - 1)
    for n in range(1, P):
        f = [0] * P
        f[0] = 1
        f[n] = -1
        for _ in range(24):
            prod = mul(prod, f)
    Delta = [0] + prod[: P - 1]
    basis = [pw(E4, 7), mul(pw(E4, 4), Delta), mul(E4, pw(Delta, 2))]
    # solve for coefficients c with sum c_i basis_i having a0=1, a1=a2=0
    A = [[Fraction(basis[i][n]) for i in range(3)] for n in range(3)]
    rhs = [Fraction(1), Fraction(0), Fraction(0)]
    # Gauss
    M = [A[r] + [rhs[r]] for r in range(3)]
    for c in range(3):
        p = next(r for r in range(c, 3) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for r in range(3):
            if r != c:
                M[r] = [x - M[r][c] * y for x, y in zip(M[r], M[c])]
    coef = [M[r][3] for r in range(3)]
    theta = [sum(coef[i] * basis[i][n] for i in range(3)) for n in range(P)]
    assert all(t.denominator == 1 for t in theta)
    theta = [int(t) for t in theta]
    # coefficient of q^n counts vectors of norm 2n
    res = {"theta_coefficients_q^0..q^7": theta,
           "norm6_count": theta[3], "norm8_count": theta[4],
           "expected_norm6": 15590400, "expected_norm8": 36957286800}
    res["verdict"] = verdict(theta[3] == 15590400 and theta[4] == 36957286800 and theta[1] == theta[2] == 0)
    log(res)
    record("theta", res)


def step_v0(args):
    G = gram()
    res = {}
    minors = bareiss_minors(G)
    res["1_integer_symmetric"] = bool(np.array_equal(G, G.T))
    res["1_diag_even"] = bool((np.diag(G) % 2 == 0).all())
    res["1_det"] = int(minors[-1]) if len(minors) == N else None
    res["1_positive_definite(leading minors > 0)"] = bool(len(minors) == N and all(m > 0 for m in minors))
    g1, g2 = generators()
    t = tau()
    I = np.eye(N, dtype=np.int64)
    res["2_g_G_gT_eq_G"] = [bool(np.array_equal(imul(imul(g, G), g.T), G)) for g in (g1, g2)]
    res["2_g_tau_eq_tau_g"] = [bool(np.array_equal(imul(g, t), imul(t, g))) for g in (g1, g2)]
    res["2_tau_G_tauT_eq_5G"] = bool(np.array_equal(imul(imul(t, G), t.T), 5 * G))
    res["2_tau^2_eq_-5"] = bool(np.array_equal(imul(t, t), -5 * I))
    orders = []
    for g in (g1, g2):
        a, k = g.copy(), 1
        while not np.array_equal(a, I):
            a = imul(a, g)
            k += 1
            assert k < 1000
        orders.append(k)
    res["2_generator_orders"] = orders
    H = build_group()
    res["3_group_order"] = int(H.shape[0])
    res["3_contains_-1"] = bool((H.reshape(H.shape[0], -1) == (-I).astype(np.int16).reshape(-1)).all(axis=1).any())
    res["3_max_abs_entry"] = int(np.abs(H).max())
    units, coeffs, nshort = find_units()
    res["4_qfminim_vectors_with_Q<=56(up to sign)"] = nshort
    res["4_units_count"] = int(len(units))
    res["4_units_commute_with_generators"] = bool(all(np.array_equal(imul(u, g), imul(g, u)) for u in units for g in (g1, g2)))
    # closed under multiplication, contains +-1
    keys = {u.tobytes() for u in units}
    res["4_units_closed_group"] = bool(all(imul(a, b).tobytes() in keys for a in units for b in units))
    res["4_units_contain_+-1"] = bool(I.tobytes() in keys and (-I).tobytes() in keys)
    uorders = []
    for u in units:
        a, k = u.copy(), 1
        while not np.array_equal(a, I):
            a = imul(a, u)
            k += 1
            assert k < 100
        uorders.append(k)
    res["4_unit_orders"] = sorted(uorders)
    res["4_unit_coefficients_mod_sign"] = units_mod_sign()[1].tolist()
    ok = (res["1_integer_symmetric"] and res["1_diag_even"] and res["1_det"] == 1
          and res["1_positive_definite(leading minors > 0)"] and all(res["2_g_G_gT_eq_G"])
          and all(res["2_g_tau_eq_tau_g"]) and res["2_tau_G_tauT_eq_5G"] and res["2_tau^2_eq_-5"]
          and res["3_group_order"] == GROUP_ORDER and res["3_contains_-1"]
          and res["4_units_count"] == 12 and res["4_units_commute_with_generators"]
          and res["4_units_closed_group"] and res["4_units_contain_+-1"])
    res["verdict"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("v0", res)


def step_v1(args):
    R = min_reps()
    res = {"n_reps": int(len(R))}
    nr = norms(R)
    res["5_all_norm_6"] = bool((nr == 6).all())
    O = Orbits()
    canon, size = O.canon(R)
    ncanon = len({c.tobytes() for c in canon})
    res["6_distinct_canonical_forms"] = ncanon
    res["6_orbit_size_histogram"] = {str(k): int(v) for k, v in zip(*np.unique(size, return_counts=True))}
    res["6_size_sum"] = int(size.sum())
    # build M/{+-1} by exact full images, check consistency with the sizes
    total = int(size.sum())
    assert total % 2 == 0
    Mh = np.empty((total // 2, N), dtype=np.int16)
    pos = 0
    for j, r in enumerate(R):
        im = O.images_full(r)
        assert (np.abs(im) < 2 ** 15).all()
        u = np.unique(im.astype(np.int16), axis=0)
        assert len(u) == size[j], (j, len(u), size[j])
        fnz = u[np.arange(len(u)), (u != 0).argmax(axis=1)]
        hu = u[fnz > 0]
        assert 2 * len(hu) == len(u)
        Mh[pos:pos + len(hu)] = hu
        pos += len(hu)
    assert pos == len(Mh)
    res["M_half_rows"] = int(len(Mh))
    # global distinctness without sorting the table: distinct 64-bit hashes imply distinct rows
    rng = np.random.default_rng(1)
    hv = rng.integers(0, 2 ** 63, size=N, dtype=np.uint64) | np.uint64(1)
    hs = np.empty(len(Mh), dtype=np.uint64)
    for a in range(0, len(Mh), 500000):
        hs[a:a + 500000] = (Mh[a:a + 500000].astype(np.int64).view(np.uint64) * hv).sum(axis=1)
    res["M_half_distinct_rows(lower bound by distinct hashes)"] = int(len(np.unique(hs)))
    del hs
    res["M_half_all_norm_6"] = bool((norms(Mh) == 6).all())
    res["M_half_max_abs_coordinate"] = int(np.abs(Mh).max())
    os.makedirs(CACHE, exist_ok=True)
    atomic_npy(os.path.join(CACHE, "min-half.npy"), Mh)
    ok = (res["5_all_norm_6"] and ncanon == 656 and res["6_size_sum"] == 15590400
          and res["M_half_rows"] == res["M_half_distinct_rows(lower bound by distinct hashes)"] == 7795200
          and res["M_half_all_norm_6"])
    res["verdict"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("v1", res)


def step_v2(args):
    rep, w0, which = odd_certs()
    R = min_reps()
    t = tau()
    res = {"n": int(len(rep)), "which_values": {str(k): int(v) for k, v in zip(*np.unique(which, return_counts=True))}}
    diff = w0 - imul(R[rep], t)
    res["7_class_ok"] = int(((diff % 2) == 0).all(axis=1).sum())
    nw = norms(w0)
    res["8_norm_histogram"] = {str(k): int(v) for k, v in zip(*np.unique(nw, return_counts=True))}
    res["8_norm_ok"] = int(np.isin(nw, [14, 18]).sum())
    im = InnerMax(threads=args.threads)
    mx, bound = im(w0)
    res["9_float32_partial_sum_bound"] = bound
    res["9_max_inner_histogram"] = {str(k): int(v) for k, v in zip(*np.unique(mx, return_counts=True))}
    res["9_ok"] = int((mx <= 6).sum())
    res["10_rep_is_permutation"] = bool(np.array_equal(np.sort(rep), np.arange(656)))
    # cross-check the float32 path by int64 on 50 of them
    sel = np.arange(0, 656, 13)
    mi = inner_max_int64(w0[sel])
    res["9_int64_crosscheck_agree"] = f"{int((mi == mx[sel]).sum())}/{len(sel)}"
    ok = (res["7_class_ok"] == 656 and res["8_norm_ok"] == 656 and res["9_ok"] == 656
          and res["10_rep_is_permutation"] and int((mi == mx[sel]).sum()) == len(sel))
    res["verdict"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("v2", res)


def even_all():
    """All even-class certificates, as array W[p, i] = certificate of pair idx = p + 6 i."""
    return np.stack([even_cert_part(p) for p in range(6)])


def step_v3cheap(args):
    X, sizes = norm8_table()
    W = even_all()  # (6, 254664, 56)
    nrep = X.shape[0]
    t = tau()
    U, Uc = units_mod_sign()
    assert len(U) == 6
    res = {"n_reps": int(nrep), "n_pairs": int(6 * nrep)}
    js = [even_cert_json(p) for p in range(6)]
    res["14_unresolved_total"] = int(sum(len(j["unresolved"]) for j in js))
    res["14_done_total"] = int(sum(j["done"] for j in js))
    # 11: (x u, x u) = 8 for all x, u, and match[p, i, k] = (w0[p,i] - x_i u_k tau in 2L)
    match = np.zeros((6, nrep, 6), dtype=bool)
    xu_norm_ok = True
    for k in range(6):
        XU = imul(X, U[k])
        xu_norm_ok &= bool((norms(XU) == 8).all())
        XUt = imul(XU, t)
        for p in range(6):
            match[p, :, k] = ((W[p] - XUt) % 2 == 0).all(axis=1)
    res["11_all_xu_norm_8"] = xu_norm_ok
    res["11_reps_norm_8"] = bool((norms(X) == 8).all())
    anym = match.any(axis=2)
    res["11_certs_in_some_class"] = int(anym.sum())
    nmatch = match.sum(axis=2)
    res["11_certs_matching_more_than_one_unit"] = int((nmatch > 1).sum())
    # perfect matching between the 6 certificates of x_i and the 6 units
    from itertools import permutations
    perms = list(permutations(range(6)))
    pm_count = np.zeros(nrep, dtype=np.int64)
    perm_used = {}
    for s in perms:
        ok = np.ones(nrep, dtype=bool)
        for p in range(6):
            ok &= match[p, :, s[p]]
        pm_count += ok
        if ok.any():
            perm_used[str(s)] = int(ok.sum())
    res["11_reps_with_perfect_matching"] = int((pm_count >= 1).sum())
    res["11_reps_with_unique_matching"] = int((pm_count == 1).sum())
    res["11_matchings_used(part p -> unit index in my order)"] = perm_used
    # 12: norms
    nw = np.stack([norms(W[p]) for p in range(6)])
    res["12_norm_histogram"] = {str(k): int(v) for k, v in zip(*np.unique(nw, return_counts=True))}
    res["12_norm_ok"] = int(np.isin(nw, [12, 16, 20]).sum())
    res["sizes_sum"] = int(sizes.sum())
    ok = (res["14_unresolved_total"] == 0 and res["14_done_total"] == NPAIRS and xu_norm_ok
          and res["11_certs_in_some_class"] == NPAIRS and res["11_reps_with_perfect_matching"] == nrep
          and res["12_norm_ok"] == NPAIRS)
    res["verdict(11,12,14)"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("v3cheap", res)


def step_v3sample(args):
    rng = np.random.default_rng(20261016)
    idx = np.sort(rng.choice(NPAIRS, size=args.n, replace=False))
    W = even_all()
    Ws = W[idx % 6, idx // 6]
    im = InnerMax(threads=args.threads)
    t0 = time.time()
    mx, bound = im(Ws)
    dt = time.time() - t0
    res = {"n": int(args.n), "seed": 20261016, "seconds": round(dt, 1),
           "seconds_per_cert": dt / args.n,
           "extrapolated_hours_for_all": dt / args.n * NPAIRS / 3600,
           "threads": args.threads, "13_float32_partial_sum_bound": bound,
           "13_max_inner_histogram": {str(k): int(v) for k, v in zip(*np.unique(mx, return_counts=True))},
           "13_ok": int((mx <= 6).sum())}
    # cross-check: plain int64 on the first args.check of them; threaded vs single on 64
    nc = min(args.check, args.n)
    t0 = time.time()
    mi = inner_max_int64(Ws[:nc])
    res["13_int64_crosscheck"] = f"{int((mi == mx[:nc]).sum())}/{nc}"
    res["13_int64_crosscheck_seconds"] = round(time.time() - t0, 1)
    im.threads = 1
    m1, _ = im(Ws[:64])
    res["13_threads_vs_single_64"] = f"{int((m1 == mx[:64]).sum())}/64"
    atomic_npz(os.path.join(OUT, "v3sample.npz"), idx=idx, mx=mx.astype(np.int8))
    ok = res["13_ok"] == args.n and int((mi == mx[:nc]).sum()) == nc and int((m1 == mx[:64]).sum()) == 64
    res["verdict(13, sample)"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("v3sample", res)


def step_v3full(args):
    parts = range(6) if args.part == "all" else [int(args.part)]
    im = InnerMax(threads=args.threads)
    for p in parts:
        st = os.path.join(OUT, f"v3full-part{p}.npz")
        W = even_cert_part(p)
        n = len(W)
        if os.path.exists(st):
            z = np.load(st)
            mx = z["mx"].copy()
            done = int(z["done"])
            secs = float(z["seconds"])
        else:
            mx = np.full(n, -1, dtype=np.int8)
            done, secs = 0, 0.0
        log(f"part {p}: resume at {done}/{n}")
        last = time.time()
        while done < n:
            t0 = time.time()
            e = min(n, done + args.chunk)
            m, bound = im(W[done:e])
            mx[done:e] = m
            secs += time.time() - t0
            done = e
            if time.time() - last > 120 or done == n:
                atomic_npz(st, mx=mx, done=np.int64(done), seconds=np.float64(secs), bound=np.int64(bound))
                last = time.time()
                log(f"part {p}: {done}/{n}, {secs / done * 1e3:.1f} ms/cert, max so far {int(mx[:done].max())}, bad {int((mx[:done] > 6).sum())}")
        log(f"part {p}: complete")
    # summary over all completed parts
    tot = {}
    alldone = True
    hist = {}
    nbad = 0
    secs = 0.0
    bounds = []
    for p in range(6):
        st = os.path.join(OUT, f"v3full-part{p}.npz")
        if not os.path.exists(st):
            alldone = False
            continue
        z = np.load(st)
        d = int(z["done"])
        if d < 254664:
            alldone = False
        m = z["mx"][:d]
        for k, v in zip(*np.unique(m, return_counts=True)):
            hist[str(k)] = hist.get(str(k), 0) + int(v)
        nbad += int((m > 6).sum())
        secs += float(z["seconds"])
        bounds.append(int(z["bound"]))
        tot[p] = d
    res = {"done_per_part": tot, "13_max_inner_histogram": hist, "13_bad": nbad,
           "cpu_wall_seconds_in_inner_products": round(secs, 1), "threads": args.threads,
           "13_float32_partial_sum_bound(last chunk of each part; every chunk asserted < 2^24)": max(bounds) if bounds else None}
    # the bound over ALL certificates, recomputed from the inputs (the per-part value above is only
    # that of the last saved chunk)
    maxM = int(np.abs(np.asarray(min_half())).max())
    l1 = max(int(np.abs(imul(even_cert_part(p), gram())).sum(axis=1).max()) for p in range(6))
    res["13_float32_partial_sum_bound(all certificates)"] = l1 * maxM
    assert l1 * maxM < F32_LIMIT
    if alldone:
        res["verdict(13, all)"] = verdict(nbad == 0 and sum(tot.values()) == NPAIRS)
    else:
        res["verdict(13, all)"] = "INCOMPLETE"
    log(json.dumps(res, indent=1))
    record("v3full", res)


def step_v4(args):
    X, sizes = norm8_table()
    U, _ = units_mod_sign()
    nrep = len(X)
    res = {"15_reps_norm_8": bool((norms(X) == 8).all()), "15_sizes_sum": int(sizes.sum())}
    st = os.path.join(OUT, "v4-state.npz")
    if os.path.exists(st):
        z = np.load(st)
        canon, osz, done = z["canon"].copy(), z["osz"].copy(), int(z["done"])
    else:
        canon = np.zeros((nrep, 6, N), dtype=np.int16)
        osz = np.zeros((nrep, 6), dtype=np.int64)
        done = 0
    O = Orbits()
    log(f"v4: resume at {done}/{nrep}")
    last = time.time()
    secs = 0.0 if done == 0 else float(np.load(st)["seconds"])
    while done < nrep:
        t0 = time.time()
        e = min(nrep, done + args.chunk)
        XU = np.concatenate([imul(X[done:e], U[k]) for k in range(6)])  # k-major
        assert (np.abs(XU) < 2 ** 15).all()
        c, s = O.canon(XU)
        assert (np.abs(c) < 2 ** 15).all()
        m = e - done
        for k in range(6):
            canon[done:e, k] = c[k * m:(k + 1) * m]
            osz[done:e, k] = s[k * m:(k + 1) * m]
        secs += time.time() - t0
        done = e
        if time.time() - last > 120 or done == nrep:
            atomic_npz(st, canon=canon, osz=osz, done=np.int64(done), seconds=np.float64(secs))
            last = time.time()
            log(f"v4: {done}/{nrep}, {secs / done * 1e3:.2f} ms/rep")
    flat = canon.reshape(-1, N)
    key = np.ascontiguousarray(flat).view(np.dtype((np.void, 2 * N))).ravel()
    uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    res["16_distinct_orbits"] = int(len(uniq))
    res["16_size_sum_over_distinct"] = int(osz.reshape(-1)[first].sum())
    # duplicates only within the same representative x_i (else two reps share a G.U-orbit)
    repidx = np.repeat(np.arange(nrep), 6)
    owner = repidx[first][inv]
    res["16_duplicates_across_different_reps"] = int((owner != repidx).sum())
    # per representative: sum of sizes of its distinct G-orbits == sizes[i]
    per = np.zeros(nrep, dtype=np.int64)
    seen_first = np.zeros(len(flat), dtype=bool)
    seen_first[first] = True
    np.add.at(per, repidx[seen_first], osz.reshape(-1)[seen_first])
    res["16_rep_size_matches_sizes_field"] = int((per == sizes).sum())
    res["16_orbit_size_histogram"] = {str(k): int(v) for k, v in zip(*np.unique(osz.reshape(-1)[first], return_counts=True))}
    res["seconds_canonical_forms"] = round(secs, 1)
    ok = (res["15_reps_norm_8"] and res["15_sizes_sum"] == 36957286800 and res["16_distinct_orbits"] == 1517130
          and res["16_size_sum_over_distinct"] == 36957286800 and res["16_duplicates_across_different_reps"] == 0
          and res["16_rep_size_matches_sizes_field"] == nrep)
    res["verdict"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("v4", res)


def check_even(w, x, U, t, im):
    """Full V3 check of a single (x, w0): returns dict of conditions."""
    w = np.asarray(w, dtype=np.int64)
    cls = any((((w - imul(imul(x, u), t)) % 2) == 0).all() for u in U)
    nw = int(norms(w[None])[0])
    mx, _ = im(w[None])
    return {"class": bool(cls), "norm": nw, "norm_ok": nw in (12, 16, 20), "max_inner": int(mx[0]),
            "inner_ok": int(mx[0]) <= 6,
            "pass": bool(cls and nw in (12, 16, 20) and int(mx[0]) <= 6)}


def step_neg(args):
    rng = np.random.default_rng(7)
    X, _ = norm8_table()
    U, _ = units_mod_sign()
    t = tau()
    G = gram()
    im = InnerMax(threads=args.threads)
    Mh = min_half()
    res = {}
    # (a) one-coordinate modifications of real certificates
    W0 = even_cert_part(0)
    picks = rng.choice(len(W0), size=args.n, replace=False)
    outcomes = {"+1": [], "+2": [], "+2y": []}
    for i in picks:
        x = X[i]
        w = W0[i].copy()
        base = check_even(w, x, U, t, im)
        assert base["pass"], ("unmodified certificate fails", int(i), base)
        j = int(rng.integers(N))
        w1 = w.copy(); w1[j] += 1
        outcomes["+1"].append(check_even(w1, x, U, t, im))
        w2 = w.copy(); w2[j] += 2
        outcomes["+2"].append(check_even(w2, x, U, t, im))
        # w + 2y for a minimal vector y with (w, y) = -3: same class, and (w + 2y, y) = 9 > 6
        # target (w,y) = -5 when N in {12,16}: then w + 2y has norm N + 4 in {16, 20} (allowed)
        # and (w + 2y, y) = 7, so only condition 13 can reject it
        nw = int(norms(w[None])[0])
        target = -5 if nw in (12, 16) else -3
        ips = im.ips(w)
        cand = np.nonzero(ips == target)[0]
        if len(cand):
            y = np.asarray(Mh[cand[rng.integers(len(cand))]], dtype=np.int64)
            w3 = w + 2 * y
            r3 = check_even(w3, x, U, t, im)
            r3["expected_max_inner_at_least"] = target + 12
            outcomes["+2y"].append(r3)
    for k, v in outcomes.items():
        res[f"a_{k}_tried"] = len(v)
        res[f"a_{k}_rejected"] = int(sum(not o["pass"] for o in v))
        res[f"a_{k}_failed_class"] = int(sum(not o["class"] for o in v))
        res[f"a_{k}_failed_norm"] = int(sum(not o["norm_ok"] for o in v))
        res[f"a_{k}_failed_inner"] = int(sum(not o["inner_ok"] for o in v))
    res["a_+2y_max_inner_values"] = sorted({o["max_inner"] for o in outcomes["+2y"]})
    # (b) gluing B (the negative control): its known collision (x, v)
    tb = _npz("negative_control_B.npz", "tau_B").astype(np.int64)
    wit = {"x": _npz("negative_control_B.npz", "x"), "v": _npz("negative_control_B.npz", "v")}
    hits = _npz("negative_control_B.npz", "hit_pairs").astype(np.int64)
    x = np.array(wit["x"], dtype=np.int64)
    v = np.array(wit["v"], dtype=np.int64)
    res["b_tauB_G_tauBT_eq_5G"] = bool(np.array_equal(imul(imul(tb, G), tb.T), 5 * G))
    res["b_x_norm"] = int(norms(x[None])[0])
    res["b_v_norm"] = int(norms(v[None])[0])
    res["b_v_in_class_x_tauB"] = bool((((v - imul(x[None], tb)[0]) % 2) == 0).all())
    res["b_hits_listed"] = int(len(hits))
    res["b_hits_norm_8"] = bool((norms(hits) == 8).all())
    # w0 = v + 2y over ALL minimal vectors y (both signs) and y = 0: none may pass 12 and 13
    ips = im.ips(v)  # (v, y) for y in M/{+-1}
    # norm(v + 2 s y) = 8 + 4 s (v,y) + 24
    n_plus = 32 + 4 * ips
    n_minus = 32 - 4 * ips
    # for w = v + 2 s y: (w, s y) = s (v, y) + 12 (since (y, y) = 6)
    ncand, npass, ipw = 0, 0, {}
    sample = []
    for s, nn in ((1, n_plus), (-1, n_minus)):
        sel = np.nonzero(np.isin(nn, [12, 16, 20]))[0]
        ncand += len(sel)
        own = s * ips[sel] + 12
        for k, c in zip(*np.unique(own, return_counts=True)):
            ipw[str(k)] = ipw.get(str(k), 0) + int(c)
        npass += int((own <= 6).sum())  # those could still pass; then need the full test
        for j in sel[:: max(1, len(sel) // 100)][:100]:
            sample.append(v + 2 * s * np.asarray(Mh[j], dtype=np.int64))
    res["b_candidates_v+2y_tried(all 15,590,400 y, plus y=0)"] = 15590400 + 1
    res["b_candidates_with_norm_in_{12,16,20}"] = int(ncand)
    res["b_their_inner_with_own_y_histogram"] = ipw
    res["b_those_not_rejected_by_own_y"] = npass
    mx, _ = im(np.array(sample))
    res["b_full_test_on_sample"] = len(sample)
    res["b_full_test_sample_passing"] = int((mx <= 6).sum())
    res["b_those_with_max_inner<=6"] = npass + int((mx <= 6).sum())
    res["b_v_itself_norm_ok"] = res["b_v_norm"] in (12, 16, 20)
    # descent attempt (independent of the producer): start from x tau_B, reduce w <- w - 2 s u
    # while the norm decreases; then test with the verifier
    desc = []
    for h in hits:
        w = imul(h[None], tb)[0]
        for _ in range(200):
            ips = im.ips(w)
            j = int(np.abs(ips).argmax())
            s = 1 if ips[j] > 0 else -1
            # norm(w - 2 s u) = n - 4 |(w,u)| + 24 decreases iff |(w,u)| > 6
            if abs(int(ips[j])) <= 6:
                break
            w = w - 2 * s * np.asarray(Mh[j], dtype=np.int64)
        nw = int(norms(w[None])[0])
        mxw = int(np.abs(im.ips(w)).max())
        desc.append({"norm": nw, "max_inner": mxw, "pass_12_13": nw in (12, 16, 20) and mxw <= 6})
    res["b_descent_results"] = desc
    res["b_descent_passes"] = int(sum(d["pass_12_13"] for d in desc))
    ok = (res["a_+1_rejected"] == res["a_+1_tried"] and res["a_+2y_rejected"] == res["a_+2y_tried"]
          and res["b_v_in_class_x_tauB"] and res["b_v_norm"] == 8 and res["b_those_with_max_inner<=6"] == 0
          and res["b_descent_passes"] == 0)
    res["verdict(negative controls behave as expected)"] = verdict(ok)
    log(json.dumps(res, indent=1))
    record("neg", res)


def step_summary(args):
    out = {}
    for s in ["v0", "theta", "v1", "v2", "v3cheap", "v3sample", "v3full", "v4", "neg"]:
        p = os.path.join(OUT, s + ".json")
        if os.path.exists(p):
            r = json.load(open(p))
            out[s] = {k: v for k, v in r.items() if k.startswith("verdict")}
            out[s]["_finished"] = r.get("_finished")
            out[s]["_wall_seconds"] = r.get("_wall_seconds")
        else:
            out[s] = "not run"
    # hashes of the inputs that were verified
    h = {}
    for f in [os.path.join(D, x) for x in ["L56_gram.txt", "tau.txt", "sl2_29_generators.txt"]] + \
             [os.path.join(C, x) for x in ["min_reps.npz", "norm8_reps.npz", "commutant.npz", "odd_bounding.npz",
                                           "even_bounding.npz", "negative_control_B.npz"]]:
        h[os.path.relpath(f, ROOT)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
    out["input_sha256"] = h
    log(json.dumps(out, indent=1))
    atomic_json(os.path.join(OUT, "summary.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--check", type=int, default=200)
    ap.add_argument("--part", default="all")
    ap.add_argument("--chunk", type=int, default=2048)  # v4: reps per checkpoint unit
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log("start", args.cmd, vars(args), "host", platform.node())
    {"v0": step_v0, "theta": step_theta, "v1": step_v1, "v2": step_v2, "v3cheap": step_v3cheap,
     "v3sample": step_v3sample, "v3full": step_v3full, "v4": step_v4, "neg": step_neg,
     "summary": step_summary}[args.cmd](args)
    log("done", args.cmd)


if __name__ == "__main__":
    main()
