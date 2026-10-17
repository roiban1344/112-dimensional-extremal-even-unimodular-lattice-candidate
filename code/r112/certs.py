"""Certificate files: the data that the verification scripts check (see certificates/MANIFEST.md).
検証用データ: 検算スクリプトが確かめるデータ（certificates/MANIFEST.md を参照）。

  min_reps.npz       reps  int16 (656, 56)       representatives of the G-orbits on L_6 (F2)
                                                 L_6 の G-軌道の代表（F2）
  centralizer.npz    units int64 (6, 56, 56)     representatives of C/{+-1} (paper §4.2)
                                                 C/{±1} の代表（論文 §4.2）
  norm8_reps.npz     reps  int16 (254664, 56)    representatives x_i of the G*-orbits on L_8 (F3)
                                                 L_8 の G*-軌道の代表 x_i（F3）
  odd_bounding.npz   w0    int16 (656, 56)       odd bounding representative of r_j tau + 2L, row j (F5a)
                                                 r_j tau + 2L の奇の下界代表元（第 j 行）（F5a）
  even_bounding.npz  w0    int16 (1527984, 56)   even bounding representative of x_i u_k tau + 2L
                                                 in row 6 i + k (F5b)
                                                 x_i u_k tau + 2L の偶の下界代表元（第 6i+k 行）（F5b）
  reduction_L.txt    U     (56 x 56 integers)    BKZ-40 basis change of L used by the enumeration (F1)
                                                 列挙（F1）に使う L の BKZ-40 の基底変換
  reduction_R.txt    U     (112 x 112 integers)  basis change from (3.1) to data/R112_gram_LLL.txt
                                                 (3.1) から data/R112_gram_LLL.txt への基底変換

The file certificates/SHA256SUMS lists the hash of every file; `load` checks it.
.npz files are read with allow_pickle=False, so loading cannot execute code.
certificates/SHA256SUMS に各ファイルのハッシュがあり、`load` が照合する。
.npz は allow_pickle=False で読むので、読み込みでコードが実行されることはない。
"""
import os

import numpy as np

from .data import read_integers, sha256_file

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "certificates")


def expected_hashes(directory):
    """Parse SHA256SUMS ("<hash>  <file>" per line). / SHA256SUMS（1 行に「ハッシュ  ファイル名」）を読む。"""
    path = os.path.join(directory, "SHA256SUMS")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    h, name = line.split()
                    out[name.lstrip("*")] = h
    return out


def path_checked(name, directory=None, require_hash=True):
    """Path of a certificate file after checking its SHA-256 against SHA256SUMS.
    SHA256SUMS と SHA-256 を照合したうえで、検証用データのファイルのパスを返す。"""
    directory = directory or DEFAULT_DIR
    path = os.path.join(directory, name)
    hashes = expected_hashes(directory)
    if name in hashes:
        got = sha256_file(path)
        if got != hashes[name]:
            raise ValueError(f"{path}: SHA-256 {got} differs from SHA256SUMS {hashes[name]}")
    elif require_hash:
        raise ValueError(f"{name} is not listed in {directory}/SHA256SUMS")
    return path


def load(name, key, directory=None, require_hash=True):
    """One array from a certificate .npz file. / 検証用データの .npz から配列を 1 つ読む。"""
    with np.load(path_checked(name, directory, require_hash), allow_pickle=False) as z:
        return z[key]


def load_matrix(name, n, directory=None, require_hash=True):
    """An n x n integer matrix from a text certificate. / テキストの検証用データから n × n の整数行列を読む。"""
    v = read_integers(path_checked(name, directory, require_hash))
    assert v.size == n * n, name
    return v.reshape(n, n)


def save_npz(path, **arrays):
    """Write an .npz atomically (temporary file, then rename). / .npz を不可分に書く（一時ファイルに書いてから改名）。"""
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def save_matrix(path, M, header):
    """Write an integer matrix as text, one row per line, after '#' header lines.
    整数行列をテキストで書く（'#' の見出し行のあと、1 行に 1 行ずつ）。"""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for line in header.splitlines():
            f.write("# " + line + "\n")
        for row in np.asarray(M):
            f.write(" ".join(str(int(v)) for v in row) + "\n")
    os.replace(tmp, path)
