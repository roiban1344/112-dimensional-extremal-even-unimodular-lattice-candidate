#!/usr/bin/env python3
"""F1: min L = 6 for the 56-dimensional lattice L, by an enumeration whose every
decision is made in exact integer arithmetic.

Release note: for the public repository only the input path (data/L56_gram.txt), the path of the
C program and the output directory (out/crosscheck_all_data/) were changed. Requires SageMath
(PARI, fpylll) for the reduction step.
公開用の注記: 公開用に変えたのは入力のパス（data/L56_gram.txt）、C のプログラムのパス、
出力先（out/crosscheck_all_data/）だけ。簡約の段に SageMath（PARI, fpylll）が要る。

Method (independent of the earlier fplll / qfminim runs):
 1. Reduce: PARI qflllgram, then fpylll BKZ (block sizes 10..40) on a rounded Cholesky basis.  The
    floating point only chooses U; any unimodular U would do, and U is checked exactly
    (G' = U G U^T is computed in integers and has det 1, so det U = +-1).
 2. Approximate Cholesky G' ~ R^T R in float64, scale by 2^k and round: an INTEGER upper triangular
    R_int.  Q(x) := |R_int x^T|^2 is an exact integer quadratic form, and
        E := 2^{2k} G' - R_int^T R_int     (exact integer matrix),
        2^{2k} q(x) = Q(x) + x E x^T,   |x E x^T| <= ||E||_2 |x|^2 <= ||E||_F ||G'^{-1}||_inf q(x)
    (|x|^2 <= q(x) / lambda_min(G') and 1/lambda_min = ||G'^{-1}||_2 <= ||G'^{-1}||_inf, G'^{-1} exact).
    Hence q(x) <= r  implies  Q(x) <= T_r := r 2^{2k} + r ||E||_F ||G'^{-1}||_inf  (rounded up).
 3. Enumerate {x : Q(x) <= T_r} exactly (C, __int128: code/crosscheck/all_data/min_L_enum.c).
    The enumeration ranges come from exact integer square roots, so nothing is pruned by rounding.
 4. Compute q(x) exactly for every enumerated x.
 r = 4: no x != 0 with q(x) <= 4  =>  min L >= 6 (L is even).
 Positive control (--test e8x7): the same pipeline on E8^7 in a random basis, where the number of
    vectors of norm 2 and 4 is known (7*240 = 1680 and 7*2160 + 21*240^2 = 1,224,720).
    (r = 6 on L, which would recount the 15,590,400 minimal vectors, needs ~5e12 nodes: not done.)

usage (from the repository root): sage -python -I code/crosscheck/all_data/min_L.py <r> [--k 40] [--test e8x7]
"""
import sys, os, json, time, subprocess, platform, argparse
from fractions import Fraction
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out", "crosscheck_all_data")
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "min_L_enum.c")
N = 56
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:9.1f}s]", *a, flush=True)


def read_gram():
    rows = [list(map(int, l.split())) for l in open(os.path.join(D, "L56_gram.txt")) if l.strip() and not l.startswith("#")]
    assert len(rows) == N and all(len(r) == N for r in rows)
    return rows


def matmul(A, B):
    return [[sum(a * b for a, b in zip(row, col)) for col in zip(*B)] for row in A]


def transpose(A):
    return [list(r) for r in zip(*A)]


def bareiss_minors(M):
    A = [list(r) for r in M]
    n = len(A)
    out, prev = [], 1
    for k in range(n):
        if A[k][k] == 0:
            return out + [0]
        out.append(A[k][k])
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                A[i][j] = (A[i][j] * A[k][k] - A[i][k] * A[k][j]) // prev
        prev = A[k][k]
    return out


def exact_inverse(M):
    n = len(M)
    A = [[Fraction(x) for x in r] + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
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
    assert all(x.denominator == 1 for r in inv for x in r)
    return [[int(x) for x in r] for r in inv]


def isqrt_ceil(v):
    from math import isqrt
    s = isqrt(v)
    return s if s * s == v else s + 1


E8 = [[2, -1, 0, 0, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0, 0, 0], [0, -1, 2, -1, 0, 0, 0, -1], [0, 0, -1, 2, -1, 0, 0, 0],
      [0, 0, 0, -1, 2, -1, 0, 0], [0, 0, 0, 0, -1, 2, -1, 0], [0, 0, 0, 0, 0, -1, 2, 0], [0, 0, -1, 0, 0, 0, 0, 2]]


def test_gram(name):
    assert name == "e8x7"
    G = [[0] * N for _ in range(N)]
    for b in range(7):
        for i in range(8):
            for j in range(8):
                G[8 * b + i][8 * b + j] = E8[i][j]
    assert bareiss_minors(E8)[-1] == 1
    # random unimodular change of basis (products of elementary matrices), to hide the block structure
    rng = np.random.default_rng(56)
    V = [[int(i == j) for j in range(N)] for i in range(N)]
    for _ in range(400):
        i, j = rng.choice(N, 2, replace=False)
        c = int(rng.integers(-2, 3))
        V[i] = [a + c * b for a, b in zip(V[i], V[j])]
    return matmul(matmul(V, G), transpose(V))


def reduce_basis(G):
    import cypari2
    from fpylll import IntegerMatrix, GSO, LLL, BKZ
    from fpylll.algorithms.bkz2 import BKZReduction
    pari = cypari2.Pari()
    Tp = pari.qflllgram(pari.matrix(N, N, [x for row in G for x in row]))
    U0 = [[int(Tp[i, j]) for i in range(N)] for j in range(N)]  # rows = LLL basis (row convention)
    G0 = matmul(matmul(U0, G), transpose(U0))
    Lc = np.linalg.cholesky(np.array(G0, dtype=np.float64))
    B = IntegerMatrix.from_matrix([[int(round(v * 2 ** 24)) for v in row] for row in Lc])
    Uf = IntegerMatrix.identity(N)
    M = GSO.Mat(B, U=Uf)
    M.update_gso()
    LLL.Reduction(M)()
    for b in (10, 20, 30, 40):
        BKZReduction(M)(BKZ.EasyParam(b, max_loops=6))
    V = [[int(Uf[i, j]) for j in range(N)] for i in range(N)]
    return matmul(V, U0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("r", type=int)
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--maxprint", type=int, default=2000000)
    ap.add_argument("--test", default=None)
    args = ap.parse_args()
    r, k = args.r, args.k
    res = {"radius_norm": r, "k": k, "lattice": args.test or "L"}
    G = test_gram(args.test) if args.test else read_gram()
    U = reduce_basis(G)
    Gp = matmul(matmul(U, G), transpose(U))
    minors = bareiss_minors(Gp)
    res["reduced_det"] = minors[-1]
    res["reduced_pos_def"] = len(minors) == N and all(m > 0 for m in minors)
    assert res["reduced_det"] == 1 and res["reduced_pos_def"]  # => det U = +-1, U unimodular
    res["reduced_diag"] = [Gp[i][i] for i in range(N)]
    # float Cholesky, scaled integer factor
    L = np.linalg.cholesky(np.array(Gp, dtype=np.float64))
    Rf = L.T  # upper triangular, Gp ~ Rf^T Rf
    S = 2 ** k
    R = [[int(round(Rf[i, j] * S)) if j >= i else 0 for j in range(N)] for i in range(N)]
    assert all(R[i][i] > 0 for i in range(N))
    RtR = matmul(transpose(R), R)
    E = [[S * S * Gp[i][j] - RtR[i][j] for j in range(N)] for i in range(N)]
    EF = isqrt_ceil(sum(e * e for row in E for e in row))
    Gi = exact_inverse(Gp)
    Ginf = max(sum(abs(x) for x in row) for row in Gi)
    T = r * S * S + r * EF * Ginf
    res["E_frobenius_ceil"] = EF
    res["Ginv_inf_norm"] = Ginf
    res["relative_slack"] = float(Fraction(EF * Ginf, S * S))
    res["T"] = str(T)
    log({k_: v for k_, v in res.items() if k_ != "reduced_diag"})
    # compile and run the exact enumerator
    exe = os.path.join(OUT, "cache", "F1-enum")
    os.makedirs(os.path.dirname(exe), exist_ok=True)
    subprocess.check_call(["gcc", "-O2", "-o", exe, SRC, "-lm"])
    inp = f"{N} {T} {args.maxprint}\n" + "\n".join(" ".join(str(v) for v in row) for row in R) + "\n"
    t0 = time.time()
    p = subprocess.run([exe], input=inp.encode(), stdout=subprocess.PIPE, check=True)  # progress goes to stderr = the log
    res["enum_seconds"] = round(time.time() - t0, 1)
    lines = p.stdout.decode().strip().split("\n")
    summ = dict(t.split("=") for t in lines[-1].split()[1:])
    res["enum_count"] = int(summ["count"])
    res["enum_nodes"] = int(summ["nodes"])
    found = [list(map(int, l.split())) for l in lines[:-1]]
    # exact norms of what was found (only the printed ones; all of them when count <= maxprint)
    norms = {}
    if found:
        X = np.array(found, dtype=np.int64)
        Gn = np.array(Gp, dtype=np.int64)
        assert int(np.abs(X).max()) ** 2 * int(np.abs(Gn).max()) * N * N < 2 ** 62  # exact int64
        q = np.einsum("ij,ij->i", X @ Gn, X)
        for a_, b_ in zip(*np.unique(q, return_counts=True)):
            norms[int(a_)] = int(b_)
    res["printed"] = len(found)
    res["exact_norms_of_printed"] = {str(a): b for a, b in sorted(norms.items())}
    if args.test == "e8x7":
        exp = {2: 1680, 4: 1224720}
        ok = res["enum_count"] == len(found) and all(norms.get(a, 0) == b for a, b in exp.items()) \
            and sum(norms.values()) == sum(exp.values())
        res["expected"] = {str(a): b for a, b in exp.items()}
        res["verdict(positive control)"] = "PASS" if ok else "FAIL"
    elif r == 4:
        res["nonzero_vectors_with_norm<=4"] = sum(b for a, b in norms.items() if a <= 4)
        ok = res["enum_count"] == len(found) and res["nonzero_vectors_with_norm<=4"] == 0
        res["verdict(min L >= 6)"] = "PASS" if ok else "FAIL"
    else:
        # compare with the verifier's table M/{+-1}: map u -> x = u U^{-1} (coordinates in the reduced basis)
        Mh = np.load(os.path.join(OUT, "cache", "min-half.npy")).astype(np.int64)
        Ui = np.array(exact_inverse(U), dtype=np.int64)
        assert np.abs(Ui).max() * np.abs(Mh).max() * N < 2 ** 62
        h2 = h4 = 0
        a = np.arange(1, 2 * N, 2, dtype=np.uint64)
        for s in range(0, len(Mh), 500000):
            X = Mh[s:s + 500000] @ Ui
            v = (X.view(np.uint64) * a).sum(axis=1)  # wraps mod 2^64
            v2 = v * v
            h2 += int((v2).sum(dtype=np.uint64)); h4 += int((v2 * v2).sum(dtype=np.uint64))
        h2 = (2 * h2) % 2 ** 64  # both signs
        h4 = (2 * h4) % 2 ** 64
        res["table_hash2"], res["table_hash4"] = h2, h4
        res["enum_hash2"], res["enum_hash4"] = int(summ["hash2"]), int(summ["hash4"])
        ok = (res["enum_count"] == 2 * len(Mh) == 15590400 and h2 == res["enum_hash2"] and h4 == res["enum_hash4"])
        res["verdict(positive control: all minimal vectors found)"] = "PASS" if ok else "FAIL"
    res["_machine"] = {"host": platform.node(), "platform": platform.platform()}
    res["_finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    log(json.dumps({k_: v for k_, v in res.items() if k_ != "reduced_diag"}, indent=1))
    path = os.path.join(OUT, f"F1-r{r}" + (f"-{args.test}" if args.test else "") + ".json")
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.rename(path + ".tmp", path)


if __name__ == "__main__":
    main()
