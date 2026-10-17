"""All vectors of norm <= B of a positive definite integral quadratic form, by the enumeration that
cannot miss vectors (verify/enum_interval.c: interval arithmetic, exact norms at the leaves).
正定値な整数二次形式のノルム B 以下の全ベクトル。取りこぼしの無い列挙（verify/enum_interval.c: 区間演算、葉では厳密なノルム）による。

Used to check that PARI's qfminim, whose documentation does not guarantee completeness, has
returned every short vector (paper §7.8, Table D2 [C24]; also the centralizer of SL_2(29) in Aut(L)).
文書に完全性の保証が無い PARI の qfminim が、短いベクトルをすべて返したことを確かめるのに使う（論文 §7.8、表 D2 の [C24]。
Aut(L) における SL_2(29) の中心化群にも）。

The caller should pass a reasonably reduced Gram matrix (the running time depends on it, the
correctness does not). Only one of each pair +-x is returned.
呼び出し側は、ある程度簡約した Gram 行列を渡すこと（実行時間はそれに依存するが、正しさは依存しない）。±x の片方だけを返す。
"""
import math
import os
import subprocess

from . import exact

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verify", "enum_interval.c")


def _interval(q):
    f = float(q)                                  # nearest rounding / 最近接丸め
    lo, hi = math.nextafter(f, -math.inf), math.nextafter(f, math.inf)
    assert lo <= q <= hi
    return lo, hi


def compile_enumerator(build_dir):
    exe = os.path.join(build_dir, "enum_interval")
    os.makedirs(build_dir, exist_ok=True)
    if not os.path.exists(exe) or os.path.getmtime(exe) < os.path.getmtime(_SRC):
        subprocess.run([os.environ.get("CC", "gcc"), "-O2", "-std=c11", "-fno-fast-math", "-ffp-contract=off",
                        "-o", exe, _SRC, "-lm"], check=True)
    return exe


def short_vectors(G, bound, build_dir, max_vectors=10 ** 7):
    """[(x, norm)] for all nonzero x (one of each +-pair) with x G x^T <= bound; G is a list of lists of
    integers (or an integer array). Returns also the number of enumeration nodes.
    x G x^T ≤ bound となる 0 でないすべての x（±の片方）と、そのノルム。列挙の節点数も返す。"""
    G = exact.to_int_matrix(G)
    n = len(G)
    D, V = exact.ldl_exact(G)
    lines = [f"{n} {int(bound)}"]
    lines += [" ".join(str(v) for v in row) for row in G]
    for d in D:
        lo, hi = _interval(d)
        lines.append(f"{lo!r} {hi!r}")
    for i in range(n):
        for j in range(i + 1, n):
            lo, hi = _interval(V[i][j])
            lines.append(f"{lo!r} {hi!r}")
    exe = compile_enumerator(build_dir)
    proc = subprocess.run([exe, str(max_vectors)], input="\n".join(lines) + "\n", stdout=subprocess.PIPE, text=True, check=True)
    out, nodes, total = [], None, 0
    for line in proc.stdout.splitlines():
        if line.startswith("vector of norm"):
            head, coords = line.split(":")
            out.append(([int(t) for t in coords.split()], int(head.split()[-1])))
        elif line.startswith("nodes"):
            nodes = int(line.split()[1])
        elif line.startswith("nonzero vectors of norm"):
            total += int(line.split(":")[1]) // 2
    assert proc.stdout.rstrip().endswith("done"), "enumerator did not finish"
    assert len(out) == total, "more vectors than max_vectors"
    return out, nodes
