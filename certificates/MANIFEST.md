# Certificates / 検証用データ

The files in this directory are the data checked by `code/verify/`. The proof of Theorem A
depends on these files and on the verification code only; how the files were found
(`code/produce/`) does not matter, because every property used is re-checked.

このディレクトリのファイルは、`code/verify/` が確かめるデータです。定理 A の証明が頼るのは、これらのファイルと検算のコードだけです。
ファイルの作り方（`code/produce/`）は関係しません。使う性質はすべて検算し直すからです。

The large `.npz` files are attached to the GitHub Release, not stored in git. `SHA256SUMS`
(in git) fixes their content; every reader in `code/r112/certs.py` checks the hash before use.
Download them with `code/verify/fetch.sh`.

大きな `.npz` は git に入れず、GitHub Release に添付します。`SHA256SUMS`（git に入っています）が中身を固定し、
`code/r112/certs.py` の読み込みは使う前にハッシュを照合します。取得は `code/verify/fetch.sh` で行います。

File names keep the letter R of an earlier draft for the 112-dimensional lattice (`outer_automorphism_R.txt`, `reduction_R.txt`; also `data/R112_gram_LLL.txt`); the paper calls it L₁₁₂.
ファイル名には、112 次元格子の以前の呼び名 R が残っています（`outer_automorphism_R.txt`、`reduction_R.txt`、`data/R112_gram_LLL.txt`）。論文では L₁₁₂ と呼びます。

All `.npz` files are plain NumPy archives without Python objects; they are read with
`numpy.load(..., allow_pickle=False)`. Vectors are integer row vectors in the basis of
`data/L56_gram.txt`.

`.npz` はすべて Python のオブジェクトを含まない NumPy の書庫で、`numpy.load(..., allow_pickle=False)` で読みます。
ベクトルは `data/L56_gram.txt` の基底に関する整数の行ベクトルです。

| file / ファイル | array / 配列 | content / 中身 | checked by / 検算 | produced by / 作成 |
|---|---|---|---|---|
| `reduction_R.txt` | 112 × 112 integers | basis change U with U (3.1) Uᵀ = `data/R112_gram_LLL.txt` / (3.1) から配布の Gram 行列への基底変換 | v0 | p6 |
| `reduction_L.txt` | 56 × 56 integers | BKZ-40 basis change of L used by the enumeration / 列挙に使う L の BKZ-40 基底変換 | v1 | p6 |
| `min_reps.npz` | `reps` int16 (656, 56) | representatives r_j of the G-orbits on L₆ (F2) / L₆ の G-軌道の代表 | v3 | p2 |
| `centralizer.npz` | `units` int64 (6, 56, 56) | representatives u_k of C/{±1} (§4.2), in the order used for `even_bounding.npz` / C/{±1} の代表（`even_bounding.npz` の並びで） | v4 | p1 |
| `norm8_reps.npz` | `reps` int16 (254664, 56) | representatives x_i of the G*-orbits on L₈ (F3) / L₈ の G*-軌道の代表 | v4 | p3 |
| `odd_bounding.npz` | `w0` int16 (656, 56) | row j: odd bounding representative of r_j τ + 2L (F5a) / 第 j 行: r_j τ + 2L の奇の下界代表元 | v5 | p4 |
| `even_bounding.npz` | `w0` int16 (1527984, 56) | row 6i + k: even bounding representative of x_i u_k τ + 2L (F5b) / 第 6i+k 行: x_i u_k τ + 2L の偶の下界代表元 | v7 | p5 |

Supporting data (not certificates for Theorem A) / 補助のデータ（定理 A の検証用データではない）:

| file / ファイル | arrays / 配列 | content / 中身 | used by / 使うもの |
|---|---|---|---|
| `commutant.npz` | `basis` int64 (8, 56, 56) | a Z-basis of M' = End_{SL2(29)}(L) / M' の Z-基底 | crosscheck/all_data, supplementary |
| `negative_control_B.npz` | `tau_class_B` int64 (56, 56) | an element of M' with τ² = −5 in the other class (gluing B) / τ² = −5 で他方の類（貼り方 B）の M' の元 | supplementary |
| | `tau_B`, `x`, `v`, `hit_pairs` | the gluing map of gluing B as found in the search (a lift modulo 2, not of square −5), a collision (x, v), and the colliding pairs / 探索で見つけた貼り方 B の写像（2 を法とした持ち上げで、2 乗は −5 ではない）、衝突 (x, v)、衝突する組 | crosscheck/all_data (negative controls) |
| `outer_automorphism_R.txt` | 112 × 112 integers | an automorphism of L₁₁₂ inducing the outer automorphism of SL₂(29) / SL₂(29) の外部自己同型を与える L₁₁₂ の自己同型 | supplementary/automorphisms |
