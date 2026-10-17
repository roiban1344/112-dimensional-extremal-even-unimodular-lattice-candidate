# 「112 次元の極値偶ユニモジュラー格子」のコード

[English](README.md)

このディレクトリには、論文で使った計算がすべて入っています。論文の表 D2 はそれを [C1]〜[C27] として並べており、
このファイルの最後の節で、各行とここのスクリプトを対応させます。

## 何が示され、何を信じる必要があるか

定理 A（L₁₁₂ の最小は 10）は、手で証明した補題 3.2・4.2（`lean/` で Lean でも検査済み）と、§4.4 の計算事実 F0〜F5 から従います。
**`verify/`** のスクリプトは、2 種類の入力から F0〜F5 を確かめます:

* `data/` の公開済みの 4 ファイル（L の Gram 行列、SL₂(29) の生成元、τ、L₁₁₂ の簡約 Gram 行列）。SHA-256 は `r112/data.py` に固定してあります。
* `certificates/` の**検証用データ**（軌道の代表、下界代表元、基底変換。`certificates/MANIFEST.md` を参照）。SHA-256 は
  `certificates/SHA256SUMS` に固定してあります。

定理 A を受け入れるために信じる必要があるのは、検算のコード（numpy だけを使う Python 約 1,500 行と C のプログラム 1 本）と、
手で証明した補題です。検証用データを作ったコード（`produce/`）を信じる必要は**ありません**。証明が使うデータの性質は、すべて `verify/` が確かめ直します。

## 構成

| ディレクトリ | 中身 |
|---|---|
| `r112/` | 共有の小さなライブラリ（numpy のみ）: データの読み込み、厳密さの確認、群、2 を法とした標準形、最小ベクトル |
| `verify/` | F0〜F5 の検算（v0〜v7）と整合性の検査（v8） |
| `produce/` | 検証用データの生成（p1〜p6）。証明の検算には不要 |
| `crosscheck/` | 検算の第 2 の実装。`verify/`・`produce/`・`r112/` のコードを読まずに、Claude を用いて別に書いた。同じデータを別のコードで確かめるので、主な実装に誤りがあれば結果の食い違いとして現れる: `all_data/`（すべての検証用データと F1 を別の方法で）、`second_impl/`（標準形、デザインの恒等式、奇の類） |
| `supplementary/` | 論文の主張のうち、定理 A の証明には使わないものの計算（貼り方の分類、自己同型群など） |

## すぐに試す

```bash
bash code/verify/fetch.sh     # 大きな検証用データを取得してハッシュを照合する
bash code/verify/quick.sh     # 約 30 分: 長い 2 段を除く全部
bash code/verify/full.sh 3    # 約 12 CPU 時間: 全部の検算（v7 は 3 スレッド）
```

リポジトリの根から実行します。各段は検査ごとに `[ok]`/`[FAIL]` を表示し、結果・ソフトウェアの版・実行時間を `out/<段>.json` に書きます。

## 検算の各段

時間は論文に使った計算機（8 コア、15 GB、ほかの計算と共用）での実測です。

| 段 | 論文 | 確かめること | 時間 | メモリ |
|---|---|---|---|---|
| `v0_inputs.py` | F0, F4, 補題 3.2 | L は偶ユニモジュラー。g₁, g₂ は G を保ち、−I を含む位数 24,360 の群を生成し、τ と可換。τ² = −5、τGτᵀ = 5G。(3.1) は偶ユニモジュラー。配布した L₁₁₂ の Gram 行列は同じ格子 | 20 秒 | 1 GB 未満 |
| `v1_min_L.py` + `enum_interval.c` | F1、§6.1 (a)、表 9a | 取りこぼしの無い区間演算の列挙で min L = 6。陽性対照 E₈、Leech、Leech ⊥ Leech ⊥ E₈ | 45 分 | 1 GB 未満 |
| `v2_theta.py` | §2.3、§2.6、付録 B | 極値テータ級数（\|L₆\|, \|L₈\|）、Siegel 平均、表 3, 5, 8, B1, B2 | 1 秒 | — |
| `v3_minimal_orbits.py` | F2、§6.2 | 656 軌道が 15,590,400 本の最小ベクトルを覆う（展開と標準形の 2 通り） | 3 分 | 2 GB |
| `v4_norm8_orbits.py` | F3、§6.4 | 1,527,984 本の x_i u_k が、\|L₈\| を覆う相異なる 1,517,130 個の正則な軌道を与える | 10 分 | 2 GB |
| `v5_odd_bounding.py` | F5a、系 5.2、表 11 | 656 個の奇の下界代表元 | 3 分 | 2 GB |
| `v6_canonical_lookup.py` | F5b、命題 6.3、表 10 | 類 x_i u_k τ + 2L はどれもノルム 8 のベクトルを含まない（標準形）。型 (6,6) も | 10 分 | 2 GB |
| `v7_even_bounding.py` | F5b、系 5.3、表 12 | 1,527,984 個の偶の下界代表元（再開可能。`--threads`） | 8 CPU 時間 | 3 GB |
| `v8_design_identities.py` | §6.3 | 球面デザインの恒等式 18 個（整合性の検査。証明には不要） | 10 分 | 2 GB |

F5b は無関係な 2 つの議論（v6 と v7）で確かめます。証明にはどちらか一方で足ります。

### 厳密さの確保

判定はすべて整数演算で行います。整数行列の積を（速さのために）浮動小数点で計算するところでは、実際の入力から、すべての部分和が 2²⁴（float32）
または 2⁵³（float64）未満の整数であることを先に確かめます。そのような積は厳密です。補助関数は `r112/exact.py` にあり、各スクリプトの冒頭に、
どの積をこの方法で計算するかを書いてあります。F1 の列挙は外向きに丸める区間演算を使います（`verify/enum_interval.c` の冒頭）。

## 動作環境

* `verify/`、`crosscheck/second_impl/`: Python 3.10 以上と numpy（Python 3.12、numpy 2.5 で確認）。v1 には 128 ビット整数をもつ C コンパイラ（gcc か clang）。ほかのパッケージは要りません。
* `produce/`、`supplementary/`、`crosscheck/all_data/`: SageMath 10（PARI/GP、GAP、fpylll を含む）。SageMath 10.9 で確認。リポジトリの根の
  `environment.yml` で conda/micromamba から入れられます。
* `lean/`: Lean 4 と Mathlib（版は `lean/lean-toolchain` と `lean/lake-manifest.json` に固定）。

## 検証用データの生成

| スクリプト | 作るもの | 時間 |
|---|---|---|
| `produce/p1_centralizer.py` | M′ と中心化群 C（`centralizer.npz`、`commutant.npz`） | 10 分 |
| `produce/p2_minimal_orbits.py` | `min_reps.npz`（閉包、§6.2） | 10 分 |
| `produce/p3_norm8_orbits.py` | `norm8_reps.npz`（最小和元、§6.4） | 1.5 時間 |
| `produce/p4_odd_descent.py` | `odd_bounding.npz`（降下、§6.6） | 3 分 |
| `produce/p5_even_descent.py` | `even_bounding.npz`（降下、§6.7。6 分割、再開可能） | 約 190 CPU 時間 |
| `produce/p6_reduction.py` | `reduction_L.txt`、`reduction_R.txt` | 1 分 |

各生成スクリプトは、出力を配布した検証用データと比べます。p4 と p5 は配布したファイルをビット単位で再現します。p2 と p3 は別の代表を選びうるので、
標準形の集合（同じ軌道か）で比べます。

## 論文の表 D2 との対応

| 表 D2 | スクリプト |
|---|---|
| [C1] | verify/v2_theta.py |
| [C2] | verify/v0_inputs.py, supplementary/gluings.py (d), produce/p1_centralizer.py |
| [C3] | supplementary/gluings.py, supplementary/nebe/gluings_m.py (a), (b) |
| [C4] | supplementary/gluing_b.py, supplementary/nebe/gluings_m.py (c), (d) |
| [C5] | verify/v1_min_L.py + verify/enum_interval.c, produce/p6_reduction.py |
| [C6] | supplementary/nebe/reduction_to_nebe.py (with nebe/lattices.py, nebe/tables.py) |
| [C7] | supplementary/float_enumerations.py svp, qfminim |
| [C8] | produce/p2_minimal_orbits.py, verify/v3_minimal_orbits.py, verify/v8_design_identities.py |
| [C9] | produce/p3_norm8_orbits.py, verify/v4_norm8_orbits.py, verify/v8_design_identities.py |
| [C10] | produce/p4_odd_descent.py, verify/v5_odd_bounding.py, crosscheck/second_impl/odd_classes.py |
| [C11] | verify/v6_canonical_lookup.py, crosscheck/second_impl/even_classes.py |
| [C12] | supplementary/float_enumerations.py cvp610, cvp88 |
| [C13] | produce/p5_even_descent.py, verify/v7_even_bounding.py |
| [C14] | supplementary/frame_stabilizer.py |
| [C15] | supplementary/sample_check.py |
| [C16] | supplementary/node_estimate.py |
| [C17] | supplementary/frames.py |
| [C18] | supplementary/fixed_sublattices.py |
| [C19] | supplementary/anatomy_b.py |
| [C20, C21] | (not yet included / 未収録) |
| [C22] | lean/ (EEUL/R112/Chain.lean) |
| [C23] | supplementary/nebe/identify_nebe.py (with nebe/lattices.py, nebe/cycloquat.py) |
| [C24] | supplementary/automorphisms/aut_normalizer.py |
| [C25] | supplementary/automorphisms/group_facts.g, out_facts.g, aut_diagonal.py |
| [C26] | supplementary/automorphisms/aut_structure.py, aut_character.py, endomorphisms.py |
| [C27] | crosscheck/all_data/check_all.py, min_L.py, min_L_enum.c |
