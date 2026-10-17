"""Reading the four input files of the paper (Appendix D).
論文の 4 つの入力ファイル（付録 D）の読み込み。

  data/L56_gram.txt          Gram matrix G of the 56-dimensional lattice L (56 x 56)
                             56 次元の格子 L の Gram 行列 G
  data/sl2_29_generators.txt two isometries g1, g2 of L generating SL_2(29)
                             SL_2(29) を生成する L の等長変換 g1, g2
  data/tau.txt               the endomorphism tau of L with tau^2 = -5
                             tau^2 = -5 を満たす L の自己準同型 tau
  data/R112_gram_LLL.txt     an LLL-reduced Gram matrix of R (112 x 112)
                             R の LLL 簡約された Gram 行列

Every reader checks the SHA-256 hash of the file before parsing it, so that all
results refer to exactly the published data.
どの読み込みも、解釈する前にファイルの SHA-256 を照合する。結果はすべて、公開したデータそのものについての主張になる。
"""
import hashlib
import os

import numpy as np

# Default location: <repository root>/data, two levels above this file.
# 既定の置き場所: このファイルの 2 つ上（リポジトリの根）の data/。
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# SHA-256 of the published files (first release of the repository, 14 October 2026).
# 公開したファイルの SHA-256（リポジトリの初版、2026 年 10 月 14 日）。
SHA256 = {
    "L56_gram.txt": "4d13bd51fcf3cb8c5706bde45828713d7eeb4d18a37e8e9d84fb2700f4bef2f0",
    "sl2_29_generators.txt": "97f88443de1fd1fdeb010ecb668dbbb03a8e0e989a18c0ad85a751687012ad90",
    "tau.txt": "5f5a82e2f3bf9de7e8e170572650a9d823c77ef4bdf344a85745dbaf0a7f4bcf",
    "R112_gram_LLL.txt": "bc899ed637f43e1a47f04926542a56804812f870095ff88eaa638ae576bb733b",
}

N = 56


def sha256_file(path):
    """SHA-256 of a file, as a hex string. / ファイルの SHA-256（16 進文字列）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_integers(path, expected_sha256=None):
    """All integers of a text file, ignoring lines that start with '#'.
    '#' で始まる行を除いた、テキストファイル中のすべての整数。"""
    if expected_sha256 is not None:
        got = sha256_file(path)
        if got != expected_sha256:
            raise ValueError(f"{path}: SHA-256 {got} differs from the published {expected_sha256}")
    values = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            values.extend(int(t) for t in line.split())
    return np.array(values, dtype=np.int64)


def _matrices(name, n, count, directory):
    v = read_integers(os.path.join(directory or DEFAULT_DIR, name), SHA256[name])
    if v.size != count * n * n:
        raise ValueError(f"{name}: expected {count} matrices of size {n} x {n}, found {v.size} integers")
    return v.reshape(count, n, n)


def gram(directory=None):
    """Gram matrix G of L, int64 (56, 56). / L の Gram 行列 G。"""
    return _matrices("L56_gram.txt", N, 1, directory)[0]


def generators(directory=None):
    """The two generators g1, g2 of SL_2(29), int64 (2, 56, 56). / SL_2(29) の 2 つの生成元。"""
    return _matrices("sl2_29_generators.txt", N, 2, directory)


def tau(directory=None):
    """The matrix tau, int64 (56, 56). / 行列 tau。"""
    return _matrices("tau.txt", N, 1, directory)[0]


def gram_R_reduced(directory=None):
    """The distributed reduced Gram matrix of R, int64 (112, 112). / 配布した R の簡約 Gram 行列。"""
    return _matrices("R112_gram_LLL.txt", 2 * N, 1, directory)[0]


def gram_R(G, t):
    """Gram matrix (3.1) of R in the basis (e_i, e_i tau)/sqrt2, (0, 2 e_i)/sqrt2:
        [[3G, tau G], [(tau G)^T, 2G]].
    基底 (e_i, e_i tau)/√2, (0, 2e_i)/√2 に関する R の Gram 行列 (3.1)。

    Derivation / 導出: with B = [[I, tau], [0, 2I]] the Gram matrix is B diag(G, G) B^T / 2
    = [[(G + tau G tau^T)/2, tau G], [G tau^T, 2G]], and tau G tau^T = 5G gives 3G.
    B = [[I, tau], [0, 2I]] として Gram 行列は B diag(G, G) B^T / 2。tau G tau^T = 5G より左上は 3G。
    (This formula is checked against B diag(G, G) B^T / 2 in verify/v0_inputs.py.)
    （この式は verify/v0_inputs.py で B diag(G, G) B^T / 2 と照合する。）
    """
    tG = t @ G
    return np.block([[3 * G, tG], [tG.T, 2 * G]])
