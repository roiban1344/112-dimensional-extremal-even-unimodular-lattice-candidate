# S10a. Finite group facts F1-F5 used in Appendix G (normality of SL_2(29) in Aut(R)); Table D2 [C25].
#   Not used in the proof of Theorem A. Requires GAP 4 with the character table library CTblLib.
#   付録 G（Aut(R) における SL_2(29) の正規性）で使う有限群の事実 F1〜F5。表 D2 の [C25]。定理 A の証明には使わない。
#   GAP 4 と指標表ライブラリ CTblLib が要る。
# Run / 実行:  gap -q code/supplementary/automorphisms/group_facts.g
# (F1) orders of the proper subgroups of PSL_2(29) of index <= 112 (expected: 29:14 and 29:7 only)
#      指数 112 以下の PSL_2(29) の真部分群の位数（29:14 と 29:7 だけのはず）
# (F2) minimal index of a proper subgroup of SL_2(29) (expected 30) / SL_2(29) の真部分群の指数の最小（30 のはず）
# (F3) character degrees of the preimages of 29:14 and 29:7 in SL_2(29), and the multiplicity of the
#      trivial character of a subgroup of order 29 in the faithful degree-14 characters (expected 0)
#      29:14, 29:7 の逆像の既約指標の次数と、14 次の忠実な指標を位数 29 の部分群に制限したときの自明指標の重複度（0 のはず）
# (F4) maximal subgroups of Ru of order divisible by 29 (expected L2(29) only); restriction of the
#      degree-28 characters of 2.Ru to 2.L2(29) over all possible class fusions (irreducible)
#      位数が 29 で割れる Ru の極大部分群（L2(29) だけのはず）。2.Ru の 28 次の指標の 2.L2(29) への制限（可能な融合写像すべてで既約）
# (F5) |SL_2(59)| and |SL_2(173)| are not divisible by |SL_2(29)| / |SL_2(29)| で割り切れないこと
P := PSL(2, 29);;
cls := ConjugacyClassesSubgroups(P);;
small := Filtered(cls, c -> Index(P, Representative(c)) <= 112 and Index(P, Representative(c)) > 1);;
Print("F1: proper subgroups of PSL(2,29) of index <= 112 (order, index, #classes): ",
      Collected(List(small, c -> [Size(Representative(c)), Index(P, Representative(c))])), "\n");
S := SL(2, 29);;
clsS := ConjugacyClassesSubgroups(S);;
Print("F2: minimal index of a proper subgroup of SL(2,29): ",
      Minimum(List(Filtered(clsS, c -> Size(Representative(c)) < Size(S)), c -> Index(S, Representative(c)))), "\n");
tS := CharacterTable(S);;
psi := Filtered(Irr(tS), x -> x[1] = 14);;
Print("F3: number of degree-14 irreducible characters of SL(2,29): ", Length(psi),
      "; faithful (value -14 at -1): ", Number(psi, x -> x[PositionProperty(ConjugacyClasses(tS), c -> Representative(c) = -One(S))] = -14), "\n");
U := SylowSubgroup(S, 29);;
for c in small do
  H := PreImage(NaturalHomomorphismByNormalSubgroup(S, Centre(S)), Representative(c));
  Print("F3: preimage of order ", Size(H), " contains a Sylow 29-subgroup: ", ForAny(ConjugacyClassesSubgroups(H), d -> Size(Representative(d)) = 29),
        "; degrees of Irr: ", Collected(List(Irr(CharacterTable(H)), x -> x[1])), "\n");
od;
tU := CharacterTable(U);;
Print("F3: multiplicity of the trivial character of a 29-subgroup in each degree-14 character: ",
      List(psi, x -> ScalarProduct(tU, RestrictedClassFunction(x, U), TrivialCharacter(tU))), "\n");
ru := CharacterTable("Ru");;
mx := Maxes(ru);;
Print("F4: maximal subgroups of Ru with order divisible by 29: ", Filtered(mx, n -> Size(CharacterTable(n)) mod 29 = 0),
      " (out of ", Length(mx), ")\n");
t2 := CharacterTable("2.Ru");;
t2l := CharacterTable("2.L2(29)");;
fusions := PossibleClassFusions(t2l, t2);;           # try all possible fusions, not the stored one / 保存された融合写像に頼らず、可能な融合写像をすべて試す
chi28 := Filtered(Irr(t2), x -> x[1] = 28);;
Print("F4: degree-28 characters of 2.Ru: ", Length(chi28), "; number of possible class fusions 2.L2(29) -> 2.Ru: ", Length(fusions), "\n");
Print("F4: restrictions to 2.L2(29), as lists of [degree, multiplicity], over all possible fusions: ",
      Set(fusions, fus -> List(chi28, x -> List(Filtered(Irr(t2l), y -> ScalarProduct(t2l, x{fus}, y) <> 0), y -> [y[1], ScalarProduct(t2l, x{fus}, y)]))), "\n");
Print("F5: |SL(2,59)| / |SL(2,29)| = ", Size(SL(2,59)) / Size(S), ", |SL(2,173)| / |SL(2,29)| = ", Size(SL(2,173)) / Size(S), "\n");
QUIT;
