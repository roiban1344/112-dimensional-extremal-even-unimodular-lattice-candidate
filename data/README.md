# Defining matrices / 格子を定義する行列

| File | Contents / 内容 |
|---|---|
| `L56_gram.txt` | Gram matrix of the 56-dimensional base lattice / 素材となる 56 次元格子 L の Gram 行列 |
| `sl2_29_generators.txt` | Two matrices acting on the base lattice / L に作用する生成元 2 個 |
| `tau.txt` | Endomorphism used in the construction / 構成に使う自己準同型 |
| `R112_gram_LLL.txt` | Reduced Gram matrix of the 112-dimensional lattice / 112 次元格子の簡約 Gram 行列 |

Vectors are integer **row vectors**. Matrices act on the right. Norm means squared length. The paper defines the construction and the corresponding Gram matrix. The basis change to `R112_gram_LLL.txt` is provided in `certificates/reduction_R.txt` and checked by `code/verify/v0_inputs.py`.

ベクトルは整数の**行ベクトル**で、行列は右から作用します。ノルムは長さの二乗です。構成と対応する Gram 行列は論文に定義します。`R112_gram_LLL.txt` への基底変換は `certificates/reduction_R.txt` にあり、`code/verify/v0_inputs.py` が検査します。

`SHA256SUMS` fixes these four files. The verifier also checks their hashes internally. Checksums identify the data; they do not prove its mathematical properties.

`SHA256SUMS` はこの 4 ファイルを固定します。検算プログラムも内部でハッシュを照合します。ハッシュはデータを同定するもので、数学的性質の証明ではありません。
