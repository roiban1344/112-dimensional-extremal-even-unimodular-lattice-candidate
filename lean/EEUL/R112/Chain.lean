/-
# 112 次元格子 L₁₁₂ の論証（計算事実を仮定とした形式化）

論文（§7.6、表 9b）の
補題 3.2（偶性の部分）・補題 4.2・系 4.3・補題 5.1（中点補題）・系 5.2・系 5.3・§4.2（対称性による帰着）と、
「F1〜F5 ⟹ 内殻ベクトルが無い」の導出を、**抽象的な設定で**証明する。

設定: `L` は加法群、`B` は `L` 上の整数値の対称双線形形式、`τ` は `B (τ x) (τ y) = 5 B x y`、`τ (τ x) = -5x` を満たす。
`L₁₁₂` のベクトル `z = (x, τx + 2y)/√2` のノルムの 2 倍は `N x + N (τ x + 2y)`（`N x = B x x`）。
主定理 `no_inner_shell` は「`(x, τx+2y) ≠ 0` なら `N x + N (τ x + 2y) ≥ 20`」、すなわち `L₁₁₂` のノルムが 10 以上。

**仮定として置くもの（計算で確かめた事実。Lean では証明していない）**:
* `Setting.min6`（F1）: 0 でないベクトルのノルムは 6 以上。`Setting.even`: ノルムは偶数。
* `Data.cover6`, `Data.cover8`（F2, F3）: ノルム 6・8 のベクトルは、対称性の群 `S` の元で代表の像になる。
* `Data.sym`（F4）: `S` の元は等長で `τ` と可換。
* `Data.bound6`, `Data.bound8`（F5）: 各代表の類に下界代表元がある（ノルムの条件と、最小ベクトルとの内積の条件）。
ユニモジュラー性（`det = 1`）は整数行列の計算なので、ここでは扱わない。

Mathlib 全体は読み込まない（メモリ節約。`leanchecker` は使わないこと）。
-/
import Mathlib.Algebra.Group.Hom.Defs
import Mathlib.Algebra.Group.Equiv.Defs
import Mathlib.Algebra.Group.Basic
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Ring
import Mathlib.Tactic.ByContra

namespace EEUL.R112

/-- 抽象的な設定: 整数値の対称双線形形式 `B` をもつ加法群 `L` と、`√-5` として働く自己準同型 `τ`。 -/
structure Setting (L : Type*) [AddCommGroup L] where
  B : L →+ L →+ ℤ
  symm : ∀ x y, B x y = B y x
  /-- 偶格子: ノルムは偶数。 -/
  even : ∀ x, B x x % 2 = 0
  /-- **F1**（仮定）: 0 でないベクトルのノルムは 6 以上。 -/
  min6 : ∀ x, x ≠ 0 → 6 ≤ B x x
  τ : L →+ L
  τ_sim : ∀ x y, B (τ x) (τ y) = 5 * B x y
  τ_sq : ∀ x, τ (τ x) = -(5 • x)

namespace Setting

variable {L : Type*} [AddCommGroup L] (S : Setting L)

/-- ノルム `N x = B x x`。 -/
def N (x : L) : ℤ := S.B x x

theorem B_add_left (a b c : L) : S.B (a + b) c = S.B a c + S.B b c := by
  rw [map_add, AddMonoidHom.add_apply]

theorem B_add_right (a b c : L) : S.B a (b + c) = S.B a b + S.B a c := by
  rw [map_add]

theorem B_neg_left (a c : L) : S.B (-a) c = -S.B a c := by
  rw [map_neg, AddMonoidHom.neg_apply]

theorem B_neg_right (a c : L) : S.B a (-c) = -S.B a c := by
  rw [map_neg]

theorem N_add (a b : L) : S.N (a + b) = S.N a + S.N b + 2 * S.B a b := by
  unfold N
  rw [B_add_left, B_add_right, B_add_right, S.symm b a]
  ring

theorem N_neg (a : L) : S.N (-a) = S.N a := by
  unfold N
  rw [B_neg_left, B_neg_right, neg_neg]

theorem N_sub (a b : L) : S.N (a - b) = S.N a + S.N b - 2 * S.B a b := by
  rw [sub_eq_add_neg, N_add, N_neg, B_neg_right]
  ring

theorem N_two (y : L) : S.N (2 • y) = 4 * S.N y := by
  rw [two_nsmul, N_add]
  unfold N
  ring

theorem B_two_right (a y : L) : S.B a (2 • y) = 2 * S.B a y := by
  rw [two_nsmul, B_add_right]
  ring

theorem N_tau (x : L) : S.N (S.τ x) = 5 * S.N x := S.τ_sim x x

/-- 0 でないベクトルの 2 倍のノルムは 24 以上。 -/
theorem N_two_ge (y : L) (hy : y ≠ 0) : 24 ≤ S.N (2 • y) := by
  rw [N_two]
  have := S.min6 y hy
  unfold N
  omega

/-- 補題 3.2 の偶性の部分: `N x + N (τ x + 2y) = 6 N x + 4 (B (τ x) y + N y)`。とくに 4 の倍数。 -/
theorem norm_sum (x y : L) :
    S.N x + S.N (S.τ x + 2 • y) = 6 * S.N x + 4 * (S.B (S.τ x) y + S.N y) := by
  rw [N_add, N_tau, N_two, B_two_right]
  ring

theorem norm_sum_mod4 (x y : L) : (S.N x + S.N (S.τ x + 2 • y)) % 4 = 0 := by
  rw [norm_sum]
  have := S.even x
  unfold N at *
  omega

/-- `τ x ∈ 2L` なら `x ∈ 2L`（補題 4.2 の証明の `v = 0` の場合）。 -/
theorem two_dvd_of_tau (x y : L) (h : S.τ x + 2 • y = 0) : x = 2 • (S.τ y - 2 • x) := by
  have h1 : S.τ x = -(2 • y) := eq_neg_of_add_eq_zero_left h
  have h2 : S.τ (S.τ x) = -(2 • S.τ y) := by rw [h1, map_neg, map_nsmul]
  rw [S.τ_sq] at h2
  have h3 : 5 • x = 2 • S.τ y := neg_injective h2
  calc x = 5 • x - 4 • x := by abel
    _ = 2 • S.τ y - 4 • x := by rw [h3]
    _ = 2 • (S.τ y - 2 • x) := by abel

/-- `τ² ≡ 1 (mod 2)`: `v = τ x + 2y` なら `x = τ v + 2 (3x - τ y)`（付録 C.4）。 -/
theorem tau_involution_mod2 (x y : L) :
    x = S.τ (S.τ x + 2 • y) + 2 • (3 • x - S.τ y) := by
  rw [map_add, map_nsmul, S.τ_sq]
  abel

end Setting

/-! ## 主定理（補題 4.2 + 系 4.3）: 四衝突条件から内殻ベクトルが無いことが出る -/

section Main

variable {L : Type*} [AddCommGroup L] (S : Setting L)

open Setting

/-- **四衝突条件 ⟹ 内殻ベクトルが無い**（系 4.3 の「⇐」）。
仮定 `h6`: 最小ベクトル `r` の類 `τ r + 2L` の元はノルム 14 以上。
仮定 `h8`: ノルム 8 の `x` の類 `τ x + 2L` の元はノルム 12 以上。
結論: `(x, τx + 2y) ≠ 0` なら `N x + N (τ x + 2y) ≥ 20`（`L₁₁₂` のノルム ≥ 10）。 -/
theorem no_inner_shell
    (h6 : ∀ r y, S.N r = 6 → 14 ≤ S.N (S.τ r + 2 • y))
    (h8 : ∀ x y, S.N x = 8 → 12 ≤ S.N (S.τ x + 2 • y))
    (x y : L) (hz : x ≠ 0 ∨ S.τ x + 2 • y ≠ 0) :
    20 ≤ S.N x + S.N (S.τ x + 2 • y) := by
  have hmod := S.norm_sum_mod4 x y
  have hex := S.even x
  have hev := S.even (S.τ x + 2 • y)
  unfold N at hmod
  by_cases hx : x = 0
  · -- x = 0: v = 2y ≠ 0
    subst hx
    have hv : S.τ 0 + 2 • y ≠ 0 := by
      rcases hz with h | h
      · exact absurd rfl h
      · exact h
    rw [map_zero, zero_add] at hv ⊢
    have hy : y ≠ 0 := by rintro rfl; simp at hv
    have := S.N_two_ge y hy
    unfold N at this ⊢
    simp only [map_zero]
    omega
  by_cases hv : S.τ x + 2 • y = 0
  · -- v = 0: x ∈ 2L, x ≠ 0
    have hxe := S.two_dvd_of_tau x y hv
    have hw : S.τ y - 2 • x ≠ 0 := by
      intro h; rw [h] at hxe; simp at hxe; exact hx hxe
    have := S.N_two_ge _ hw
    rw [← hxe] at this
    rw [hv]
    unfold N at this ⊢
    simp only [map_zero]
    omega
  -- 両方 0 でない
  have hx6 := S.min6 x hx
  have hv6 := S.min6 _ hv
  by_contra hlt
  unfold N at hlt
  -- 和は 12 か 16。型ごとに場合分け
  have hcases : (S.B x x = 6 ∧ (S.B (S.τ x + 2 • y) (S.τ x + 2 • y) = 6 ∨ S.B (S.τ x + 2 • y) (S.τ x + 2 • y) = 10))
      ∨ (S.B x x = 8 ∧ S.B (S.τ x + 2 • y) (S.τ x + 2 • y) = 8)
      ∨ (S.B x x = 10 ∧ S.B (S.τ x + 2 • y) (S.τ x + 2 • y) = 6) := by omega
  rcases hcases with ⟨h1, h2⟩ | ⟨h1, h2⟩ | ⟨h1, h2⟩
  · have := h6 x y h1; unfold N at this; omega
  · have := h8 x y h1; unfold N at this; omega
  · -- (10,6) 型: x は v の類 τ v + 2L に入る
    have := h6 (S.τ x + 2 • y) (3 • x - S.τ y) h2
    rw [← S.tau_involution_mod2 x y] at this
    unfold N at this
    omega

end Main

/-! ## 中点補題と下界代表元（補題 5.1、系 5.2、系 5.3） -/

section Bounding

variable {L : Type*} [AddCommGroup L] (S : Setting L)

open Setting

/-- 下界代表元の条件: すべての最小ベクトル `u` について `|B w0 u| ≤ 6`。 -/
def SmallIP (w0 : L) : Prop := ∀ u, S.N u = 6 → -6 ≤ S.B w0 u ∧ S.B w0 u ≤ 6

/-- 中点の計算: `w = w0 + 2y` について、2 つの中点 `-y` と `w0 + y` のノルム。 -/
theorem midpoints (w0 y : L) :
    4 * S.N y = S.N (w0 + 2 • y) - S.N w0 - 4 * S.B w0 y ∧
    4 * S.N (w0 + y) = 3 * S.N w0 + S.N (w0 + 2 • y) + 4 * S.B w0 y := by
  constructor
  · rw [N_add, N_two, B_two_right]; ring
  · rw [N_add, N_add, N_two, B_two_right]; ring

/-- **系 5.2（奇の類の下界）**: ノルム 14 か 18 の下界代表元 `w0` があれば、その類の元はすべてノルム 14 以上。 -/
theorem odd_bound (w0 : L) (hN : S.N w0 = 14 ∨ S.N w0 = 18) (hip : SmallIP S w0) (y : L) :
    14 ≤ S.N (w0 + 2 • y) := by
  obtain ⟨e1, e2⟩ := midpoints S w0 y
  have ey := S.even y
  have ewy := S.even (w0 + y)
  have ew := S.even (w0 + 2 • y)
  have hmod : (S.N (w0 + 2 • y) - S.N w0) % 4 = 0 := by
    rw [N_add, N_two, B_two_right]; omega
  by_contra hlt
  -- y ≠ 0 かつ w0 + y ≠ 0（そうでなければ w = ±w0 でノルムが等しい）
  have hy : y ≠ 0 := by
    rintro rfl; simp at hlt; omega
  have hwy : w0 + y ≠ 0 := by
    intro h
    have : w0 + 2 • y = -w0 := by
      rw [two_nsmul, ← add_assoc, h, zero_add]; exact (eq_neg_of_add_eq_zero_right h)
    rw [this, N_neg] at hlt; omega
  have m1 := S.min6 y hy
  have m2 := S.min6 (w0 + y) hwy
  unfold N at *
  -- 最小ベクトルになる中点について内積の条件を使う
  by_cases c1 : S.B y y = 6
  · have := hip (-y) (by unfold N; rw [B_neg_left, B_neg_right, neg_neg]; exact c1)
    rw [B_neg_right] at this
    by_cases c2 : S.B (w0 + y) (w0 + y) = 6
    · have h2 := hip (w0 + y) c2
      rw [B_add_right] at h2
      rcases hN with h | h <;> omega
    · rcases hN with h | h <;> omega
  · by_cases c2 : S.B (w0 + y) (w0 + y) = 6
    · have h2 := hip (w0 + y) c2
      rw [B_add_right] at h2
      rcases hN with h | h <;> omega
    · rcases hN with h | h <;> omega

/-- **系 5.3（偶の類の下界）**: ノルム 12, 16, 20 の下界代表元 `w0` があれば、その類の元はすべてノルム 12 以上。 -/
theorem even_bound (w0 : L) (hN : S.N w0 = 12 ∨ S.N w0 = 16 ∨ S.N w0 = 20) (hip : SmallIP S w0) (y : L) :
    12 ≤ S.N (w0 + 2 • y) := by
  obtain ⟨e1, e2⟩ := midpoints S w0 y
  have ey := S.even y
  have ewy := S.even (w0 + y)
  have ew := S.even (w0 + 2 • y)
  have hmod : (S.N (w0 + 2 • y) - S.N w0) % 4 = 0 := by
    rw [N_add, N_two, B_two_right]; omega
  by_contra hlt
  have hy : y ≠ 0 := by
    rintro rfl; simp at hlt; omega
  have hwy : w0 + y ≠ 0 := by
    intro h
    have : w0 + 2 • y = -w0 := by
      rw [two_nsmul, ← add_assoc, h, zero_add]; exact (eq_neg_of_add_eq_zero_right h)
    rw [this, N_neg] at hlt; omega
  -- w = w0 + 2y が 0 の場合: N w0 = 4 N y で、N y は偶数かつ 6 以上なので N w0 ≥ 24
  have m1 := S.min6 y hy
  have m2 := S.min6 (w0 + y) hwy
  unfold N at *
  by_cases c1 : S.B y y = 6
  · have := hip (-y) (by unfold N; rw [B_neg_left, B_neg_right, neg_neg]; exact c1)
    rw [B_neg_right] at this
    by_cases c2 : S.B (w0 + y) (w0 + y) = 6
    · have h2 := hip (w0 + y) c2
      rw [B_add_right] at h2
      rcases hN with h | h | h <;> omega
    · rcases hN with h | h | h <;> omega
  · by_cases c2 : S.B (w0 + y) (w0 + y) = 6
    · have h2 := hip (w0 + y) c2
      rw [B_add_right] at h2
      rcases hN with h | h | h <;> omega
    · rcases hN with h | h | h <;> omega

end Bounding

/-! ## 対称性による帰着と、計算事実 F1〜F5 からの主定理 -/

section Reduction

variable {L : Type*} [AddCommGroup L]

/-- 計算で得たデータと、計算で確かめた事実（F2〜F5）。`Sym` は対称性の群、`reps6`・`reps8` は軌道代表。 -/
structure Data (S : Setting L) where
  Sym : Set (L ≃+ L)
  reps6 : Set L
  reps8 : Set L
  /-- **F4**: 対称性の元は等長で、`τ` と可換。 -/
  sym_iso : ∀ g ∈ Sym, ∀ a b, S.B (g a) (g b) = S.B a b
  sym_tau : ∀ g ∈ Sym, ∀ a, S.τ (g a) = g (S.τ a)
  /-- **F2**: ノルム 6 のベクトルは代表の像。 -/
  cover6 : ∀ x, S.N x = 6 → ∃ g ∈ Sym, ∃ r ∈ reps6, x = g r
  /-- **F3**: ノルム 8 のベクトルは代表の像（代表は (軌道代表) × (中心化群の元) の全体）。 -/
  cover8 : ∀ x, S.N x = 8 → ∃ g ∈ Sym, ∃ r ∈ reps8, x = g r
  /-- **F5a**: 各代表 `r ∈ reps6` の類 `τ r + 2L` に奇の下界代表元がある。 -/
  bound6 : ∀ r ∈ reps6, ∃ w0 y0, w0 = S.τ r + 2 • y0 ∧ (S.N w0 = 14 ∨ S.N w0 = 18) ∧ SmallIP S w0
  /-- **F5b**: 各代表 `r ∈ reps8` の類 `τ r + 2L` に偶の下界代表元がある。 -/
  bound8 : ∀ r ∈ reps8, ∃ w0 y0, w0 = S.τ r + 2 • y0 ∧ (S.N w0 = 12 ∨ S.N w0 = 16 ∨ S.N w0 = 20) ∧ SmallIP S w0

variable {S : Setting L} (D : Data S)

open Setting

/-- 代表の類についての下界を、群の像に移す（§4.2）。 -/
theorem class_bound_transport (g : L ≃+ L) (hg : g ∈ D.Sym) (r : L) (c : ℤ)
    (hr : ∀ y, c ≤ S.N (S.τ r + 2 • y)) (y : L) : c ≤ S.N (S.τ (g r) + 2 • y) := by
  have : S.τ (g r) + 2 • y = g (S.τ r + 2 • g.symm y) := by
    rw [map_add, map_nsmul, AddEquiv.apply_symm_apply, D.sym_tau g hg]
  rw [this]
  have := hr (g.symm y)
  unfold N at this ⊢
  rwa [D.sym_iso g hg]

/-- 下界代表元から、類全体の下界へ。 -/
theorem class_bound_of_rep (r w0 y0 : L) (hw : w0 = S.τ r + 2 • y0) (c : ℤ)
    (hb : ∀ y, c ≤ S.N (w0 + 2 • y)) (y : L) : c ≤ S.N (S.τ r + 2 • y) := by
  have : S.τ r + 2 • y = w0 + 2 • (y - y0) := by rw [hw]; abel
  rw [this]; exact hb _

include D in
/-- **定理 A の論理部分**: F1（`Setting.min6`）と F2〜F5（`Data`）を仮定すれば、
`(x, τx+2y) ≠ 0` のとき `N x + N (τ x + 2y) ≥ 20`。すなわち `L₁₁₂` に内殻ベクトルは無い。 -/
theorem theoremA (x y : L) (hz : x ≠ 0 ∨ S.τ x + 2 • y ≠ 0) :
    20 ≤ S.N x + S.N (S.τ x + 2 • y) := by
  apply no_inner_shell S _ _ x y hz
  · intro r y hr
    obtain ⟨g, hg, r0, hr0, rfl⟩ := D.cover6 r hr
    obtain ⟨w0, y0, hw, hN, hip⟩ := D.bound6 r0 hr0
    exact class_bound_transport D g hg r0 14
      (class_bound_of_rep r0 w0 y0 hw 14 (odd_bound S w0 hN hip)) y
  · intro x y hx
    obtain ⟨g, hg, r0, hr0, rfl⟩ := D.cover8 x hx
    obtain ⟨w0, y0, hw, hN, hip⟩ := D.bound8 r0 hr0
    exact class_bound_transport D g hg r0 12
      (class_bound_of_rep r0 w0 y0 hw 12 (even_bound S w0 hN hip)) y

end Reduction

end EEUL.R112
