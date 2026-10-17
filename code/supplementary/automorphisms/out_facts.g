# S10b. Finite group facts F6-F12 for the argument of Appendix G.4 that avoids the Schreier conjecture
#   (29 does not divide |Out(T)| for the simple groups T of the Hiss-Malle tables); Table D2 [C25].
#   Not used in the proof of Theorem A. Requires GAP 4 with CTblLib.
#   Schreier 予想を使わない付録 G.4 の議論のための有限群の事実 F6〜F12（Hiss-Malle の表の単純群 T について 29 は |Out(T)| を割らない）。
#   表 D2 の [C25]。定理 A の証明には使わない。GAP 4 と CTblLib が要る。
# Run / 実行:  gap -q code/supplementary/automorphisms/out_facts.g
#
# F6 (groups with 29 not dividing |T|): an automorphism alpha of order 29 has an alpha-invariant Sylow
#   p-subgroup P for each p (the number of Sylow subgroups divides |T|, prime to 29); if alpha != 1 it
#   acts nontrivially on the Frattini quotient of some P (dimension d(P) <= a_p), so d(P) >= ord_29(p).
#   Hence if a_p < ord_29(p) for all p, T has no automorphism of order 29. Cases of equality are listed.
#   F6（29 ∤ |T| の群）: 位数 29 の自己同型 alpha は各 p について alpha-不変な Sylow p-部分群 P をもつ（Sylow の個数は |T| を割るので 29 と素）。
#   alpha ≠ 1 ならある P の Frattini 商（次元 d(P) ≤ a_p）に非自明に働くので d(P) ≥ ord_29(p)。よってすべての p で a_p < ord_29(p) なら
#   T は位数 29 の自己同型をもたない。等号の場合は列挙して個別に扱う。
# F7: |Aut(PSL_2(p))| = 2 |PSL_2(p)| for p = 29, 59, 173.
# F8 (Ru): centralizer orders of elements of order 29 (gives |N_Ru(P)| = 406). The congruence of element
#   counts modulo 29 is recorded only to show that it cannot exclude anything (both sides vanish).
#   F8（Ru）: 位数 29 の元の中心化群の位数（|N_Ru(P)| = 406）。元の個数の 29 を法とした比較は、何も排除できない（両辺とも 0）ことの記録。
# F9: U8(2) = SU8(2) contains SU4(2), whose Sylow 2-subgroup is not abelian.
#   U8(2) は SU4(2) を含み、その Sylow 2-部分群は可換でない。
# F10: the nontrivial irreducible characters of Ru have degree > 112; the faithful irreducible characters of
#   2.Ru of degree <= 112 are two of degree 28 with field of values Q(i).
#   Ru の非自明な既約指標の次数は 112 を超える。2.Ru の次数 112 以下の忠実な既約指標は 28 次の 2 つで、値の体は Q(i)。
# F11: every degree-28 irreducible character of SL_2(29) restricts to the Borel subgroup (order 812) as a
#   sum of two different degree-14 irreducibles.
#   SL_2(29) の 28 次の既約指標はどれも、Borel 部分群（位数 812）への制限が異なる 14 次の既約指標 2 つの和。
# F12: 2.Ru has no subgroup isomorphic to 2 x L2(29) (no possible class fusion), so the preimage of L2(29)
#   is SL_2(29).
#   2.Ru は 2 × L2(29) と同型な部分群をもたない（可能な融合写像が 0 個）。よって L2(29) の逆像は SL_2(29)。
ord29 := p -> OrderMod(p, 29);
test := function(name, order)
  local bad, p, a;
  if order mod 29 = 0 then return [name, "29 divides |T|"]; fi;
  bad := [];
  for p in Set(Factors(order)) do
    a := Number(Factors(order), x -> x = p);
    if a >= ord29(p) then Add(bad, [p, a, ord29(p)]); fi;
  od;
  return [name, bad];
end;
res := [];
# Hiss–Malle 表 3 の次数 ≤ 112 の行に現れる単純群（CTblLib の名前。表を機械的に抜き出したもの + 正誤表の L3(3)）
names := ["M22","L3(4)","U4(3)","2F4(2)'","A10","A11","A12","A13","A14","A15","A16","A17","A6","A7","A8","A9","Co1","Co2","Co3","Fi22","F4(2)","G2(4)","G2(3)","HS","J1","J2","J3","L4(3)","M12","O8+(2)","O8-(2)","S4(5)","S4(7)","S4(9)","S4(4)","S6(2)","S6(3)","S6(5)","S8(3)","S8(2)","Suz","Sz(8)","U4(2)","U4(5)","U6(2)","3D4(2)","L3(7)","McL","O7(3)","ON","U3(11)","U3(5)","U3(8)","U3(3)","U3(4)","U3(7)","U3(9)","U4(4)","U5(2)","U5(3)","U7(2)","L3(5)","L3(8)","L3(9)","L4(4)","L5(2)","L6(2)","Ly","He","M11","M23","M24","L3(3)"];
for n in names do Add(res, test(n, Size(CharacterTable(n)))); od;
# CTblLib に無い 3 つは位数の公式で
Add(res, test("S4(11)", 11^4*(11^2-1)*(11^4-1)/2));
Add(res, test("S4(13)", 13^4*(13^2-1)*(13^4-1)/2));
Add(res, test("U8(2)", 2^28*Product([2..8], i -> 2^i - (-1)^i)/Gcd(8, 3)));
# 表 2 の系列: A_n（5 ≤ n ≤ 28; n ≥ 29 は 29 | |A_n|）、L_2(q)（4 ≤ q ≤ 225 の素数冪。次数 (q-1)/2 ≤ 112）
for n in [5..28] do Add(res, test(Concatenation("A", String(n)), Factorial(n)/2)); od;
for q in Filtered([4..225], IsPrimePowerInt) do
  Add(res, test(Concatenation("L2(", String(q), ")"), q*(q^2-1)/Gcd(2, q-1)));
od;
Print("F6: groups tested: ", Length(res), "\n");
Print("F6: groups with 29 | |T| (treated separately): ", List(Filtered(res, r -> r[2] = "29 divides |T|"), r -> r[1]), "\n");
Print("F6: groups failing a_p < ord_29(p) for some p (listed as [p, a_p, ord_29(p)]): ",
      Filtered(res, r -> IsList(r[2]) and not IsString(r[2]) and r[2] <> []), "\n");
# F7
for p in [29, 59, 173] do
  G := PSL(2, p);
  Print("F7: |Aut(PSL(2,", p, "))| / |PSL(2,", p, ")| = ", Size(AutomorphismGroup(G)) / Size(G), "\n");
od;
# F8
ru := CharacterTable("Ru");; l := CharacterTable("L2(29)");;
o29 := Filtered([1..NrConjugacyClasses(ru)], i -> OrdersClassRepresentatives(ru)[i] = 29);
Print("F8: Ru: classes of elements of order 29: ", Length(o29), ", centralizer orders ", SizesCentralizers(ru){o29},
      "; |N_Ru(P)| = 29 * 28 / (number of classes) = ", 29*28/Length(o29), "\n");
cnt := function(t, n) return Sum(Filtered([1..NrConjugacyClasses(t)], i -> OrdersClassRepresentatives(t)[i] = n), i -> SizesConjugacyClasses(t)[i]); end;
ords := Set(OrdersClassRepresentatives(ru));
Print("F8: n, #(order n in Ru) mod 29, #(order n in L2(29)) mod 29:\n");
for n in ords do Print("    ", n, ": ", cnt(ru, n) mod 29, " ", cnt(l, n) mod 29, "\n"); od;
Print("F8: orders n where the congruence fails: ", Filtered(ords, n -> (cnt(ru, n) - cnt(l, n)) mod 29 <> 0), "\n");
Print("F8: multiplicities of character degrees of Ru (max): ", Maximum(List(Collected(List(Irr(ru), x -> x[1])), c -> c[2])), "\n");
Print("F8: |PSL(2,29)| / |N_{PSL(2,29)}(P)| = number of Sylow 29-subgroups = ", Size(PSL(2,29)) / Size(Normalizer(PSL(2,29), SylowSubgroup(PSL(2,29), 29))), "\n");
su4 := SU(4, 2);;
Print("F9: |SU(4,2)| = ", Size(su4), ", Sylow 2-subgroup abelian: ", IsAbelian(SylowSubgroup(su4, 2)), ", Gcd(8, 2+1) = ", Gcd(8, 3), "\n");
Print("F10: Ru: minimal nontrivial degree ", Minimum(List(Filtered(Irr(ru), x -> x[1] > 1), x -> x[1])), "\n");
t2 := CharacterTable("2.Ru");;
z := First([1..NrConjugacyClasses(t2)], i -> OrdersClassRepresentatives(t2)[i] = 2 and SizesConjugacyClasses(t2)[i] = 1);;
fl := Filtered(Irr(t2), x -> x[z] = -x[1] and x[1] <= 112);;
Print("F10: 2.Ru: faithful irreducibles of degree <= 112: degrees ", List(fl, x -> x[1]), ", fields of values ", List(fl, x -> Field(Rationals, x)), "\n");
S := SL(2, 29);; B := Normalizer(S, SylowSubgroup(S, 29));; tS := CharacterTable(S);; tB := CharacterTable(B);;
c28 := Filtered(Irr(tS), x -> x[1] = 28);;
Print("F11: |B| = ", Size(B), "; number of degree-28 irreducibles of SL(2,29): ", Length(c28), "; their restrictions to B as [degree, multiplicity] lists: ",
      Set(c28, x -> List(Filtered(Irr(tB), y -> ScalarProduct(tB, RestrictedClassFunction(x, B), y) <> 0), y -> [y[1], ScalarProduct(tB, RestrictedClassFunction(x, B), y)])), "\n");
dp := CharacterTableDirectProduct(CharacterTable("Cyclic", 2), CharacterTable("L2(29)"));;
Print("F12: possible class fusions 2 x L2(29) -> 2.Ru: ", Length(PossibleClassFusions(dp, t2)), "; 2.L2(29) -> 2.Ru: ", Length(PossibleClassFusions(CharacterTable("2.L2(29)"), t2)), "\n");
QUIT;
