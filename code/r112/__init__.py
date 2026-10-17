"""Small shared library for the code accompanying the paper
"An extremal even unimodular lattice in dimension 112".

論文「112 次元の極値偶ユニモジュラー格子」に付けるコードの、共有の小さなライブラリ。

Modules / モジュール:
  data    read the four input files and check their SHA-256 hashes
          4 つの入力ファイルを読み、SHA-256 を照合する
  exact   exactness guards for floating-point products, exact rational/integer linear algebra
          浮動小数点の行列積が厳密であることの確認、有理数・整数の厳密な線形代数
  group   enumerate the 12,180 elements of SL_2(29)/{+-1} from the two generators
          2 つの生成元から SL_2(29)/{±1} の 12,180 元を列挙する
  mod2    classes modulo 2L as 56-bit integers, and their canonical forms
          2L を法とする類（56 ビットの整数）と、その標準形
  shells  inner products with all minimal vectors of L
          L のすべての最小ベクトルとの内積
  certs   reading and writing certificate files (.npz without pickle)
          検証用データ（pickle を使わない .npz）の読み書き

Only numpy and the Python standard library are used.
numpy と Python の標準ライブラリだけを使う。

Conventions (as in the paper, §2.1) / 約束（論文 §2.1 と同じ）:
  * vectors are integer ROW vectors; a matrix M acts by x -> x M;
    ベクトルは整数の行ベクトル。行列 M は x -> x M で作用する。
  * <x, y> = x G y^T; "norm" means <x, x> (not its square root);
    <x, y> = x G y^T。「ノルム」は <x, x>（平方根ではない）。
  * g is an isometry of L iff g G g^T = G.
    g が L の等長変換 ⟺ g G g^T = G。
"""
