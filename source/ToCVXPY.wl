(* ::Package:: *)

BeginPackage["ToCVXPY`"]


Print["ToCVXPY beta 1.2.3"]
Print["------------------"]
Print["Use GenerateTask[target, allVars, sdpMatrix, loopEquations, para, lambda, eps, name]"]
Print["to create name.json in order to transport the SDP question into CVXPY"]
Print["Parameter: "]
Print["  target: The coefficients of the target function."]
Print["          {1, 0, 2} will give allVars[1] + 2 allVars[3] as target."]
Print["  allVars: All the variables shown in this SDP problem. "]
Print["  sdpMatrix: SDP Matrix as constrains. "]
Print["  loopEquations: Equations constrains. Can include a parameter."]
Print["  para: The parameter shown in loopEquations. "]
Print["  lambda: The discrete values of para. "]
Print["  eps: The threshold of the SDP algorithm. It must be in the interval [\!\(\*SuperscriptBox[\(10\), \(-9\)]\), \!\(\*SuperscriptBox[\(10\), \(-3\)]\)]"]
Print["  name: Name of the task. Task will be saved to 'name.json'. Task as default."]
Print["------------------"]
Print["DREAM @ 20241023"]


GenerateTask::usage = "GenerateTask[target, allVars, sdpMatrix, loopEquations, para, lambda, eps, name]"


Begin["`Private`"]


(* ::Section::Closed:: *)
(*\:53d8\:91cf\:6620\:5c04*)


(* ::Text:: *)
(*\:5b9a\:4e49 CVXPY \:4e2d\:5c06\:4f1a\:4f7f\:7528\:5230\:7684\:53d8\:91cf\:ff0c\:8fd9\:91cc\:4f1a\:7ed9\:5b83\:4eec\:4e00\:4e2a\:65b0\:7684\:7b26\:53f7 w[i]\:ff0c\:53c2\:6570\:4f1a\:53d8\:6210 g[i]*)


CVXVarMap[vars_,paras_]:=Dispatch[Append[Table[vars[[i]]->w[i - 1],{i,1,Length[vars]}],paras->g]//Flatten]


(* ::Section::Closed:: *)
(*\:77e9\:9635\:4e0a\:4e09\:89d2\:5316*)


(* ::Text:: *)
(*\:8f93\:51fa\:683c\:5f0f\:5316\:540e\:7684\:77e9\:9635\:ff0c\:5176\:683c\:5f0f\:4e3a*)
(*{ Subscript[Index, 1] \[RightArrow] {Subscript[Row, 1], Subscript[Col, 2]}, Subscript[Index, 2] \[RightArrow] {Subscript[Row, 2], Subscript[Col, 2]}, ..., Cons \[RightArrow] {Row, Col, Value}}*)


(* ::Text:: *)
(*Subscript[Index, i] \:8868\:793a w[Subscript[Index, i]] \:4e2d\:7684\:6307\:6807\:ff0c\:5982 w[0] \:7684 Index \:6307\:7684\:662f 0\:3002Subscript[Row, i], Subscript[Col, i] \:4e3a\:4e24\:4e2a List\:ff0c\:8868\:793a Row[j], Col[j] \:7684\:5143\:7d20\:4e3a w[Subscript[Index, i]]\:3002*)
(*Cons \:8868\:793a\:4e86\:5e38\:6570\:7684\:4f4d\:7f6e\:ff0c\:8868\:793a Row, Col \:7684\:5143\:7d20\:4e3a Val*)


MatrixUpper[mat_,varmap_]:=Module[{upper, constant, allIndex,indexPos={}},
upper = UpperTriangularize[mat/.varmap];
(* \:6784\:9020 Rule *)
constant = mat/.varmap /. w[x_]:>0;
constant = Transpose[ArrayRules[constant][[;;-2]]/.Rule[{x_,y_},val_]:>{x - 1, y - 1, val}];
(* \:6784\:9020 Index *)
allIndex = Cases[Variables[upper],w[i_]]/.w[x_]:>x;
indexPos = Table[ToString[allIndex[[i]]]->Transpose[Position[upper,w[allIndex[[i]]]]-1],{i,1,Length[allIndex]}];
(* \:8f93\:51fa\:7ed3\:679c *)
If[Length[constant]==0, indexPos,Append[indexPos,"Cons"-> constant] ]
]


(* ::Subsection::Closed:: *)
(*\:7cfb\:6570\:89c4\:6574\:5316*)


(* ::Text:: *)
(*\:5f62\:5982 1/\[Lambda]^2-3\[Lambda] + 4 \:4f1a\:7ed9\:51fa {{1, -2}, {4, 0}, {-3, 1}} \:7684\:5f62\:5f0f*)


CoeffFormat[expr_, para_]:=Module[{ls, exps, coef, const = 0, constPos, res = {}},
ls = If[Head[expr] === Plus, List @@ expr, {expr}];
exps = Exponent[#, para] &/@ ls /. -Infinity->0;
(* \:7b5b\:9009\:5e38\:6570\:9879\:ff0c\:5e76\:5c06\:5176\:4ece\:8868\:8fbe\:5f0f\:4e2d\:5220\:53bb *)
constPos = Position[exps, 0]//Flatten;
If[Length[constPos] =!= 0, 
res = {{ls[[constPos[[1]]]], 0}};ls = Delete[ls, constPos];exps = Delete[exps, constPos];
];
(* \:63d0\:53d6\:7cfb\:6570 *)
coef = Coefficient[#[[1]], \[Lambda]^#[[2]]]& /@ Transpose[{ls, exps}];
(* \:603b\:7ed3\:7ed3\:679c *)
Join[res, Transpose[{coef, exps}]]
]
SetAttributes[CoeffFormat, Listable]


(* ::Subsection::Closed:: *)
(*\:79bb\:6563\:77e9\:9635\:751f\:6210*)


GenSparseMatrix[arrayRules_, para_]:=Module[{index, row = {}, col = {}, value = {}},
index = arrayRules[[1]]/.Rule[{rw_, cl_, id_}, val_]:>id;
{row, col, value} = Transpose[arrayRules /.Rule[{rw_, cl_, id_}, val_]:>{rw-1,cl-1,val}];
index -> {row, col, (*CoeffFormat[*)value}
]


(* ::Subsection:: *)
(*\:652f\:6301\:5173\:8054\:77e9\:9635\:5143\:662f w[i] \:7ebf\:6027\:7ec4\:5408\:7684\:62c6\:89e3*)


SuperMatrixUpper[mat_, varmap_, para_]:=Module[{upper, allIndex, consMat, indexMat, indexSort},
upper = UpperTriangularize[mat/.varmap]//Expand;
(* \:77e9\:9635\:77e2\:91cf\:5316 *)
(* \:63d0\:53d6\:7cfb\:6570\:77e9\:9635 *)
allIndex = Cases[Variables[upper], w[i_]] /. w[x_] :> x;
{consMat, indexMat} = CoefficientArrays[upper, w /@ allIndex] // Normal;
(* \:5904\:7406 indexMat\:ff0c\:63d0\:53d6\:7cfb\:6570 *)
indexMat = DeleteCases[ArrayRules[indexMat], Rule[x_, 0]];
indexMat = SortBy[GatherBy[indexMat, #/.Rule[x_,y_]:>x[[-1]]&], #[[1]]/.Rule[{x_, y_, z_}, w_]:>z&];
indexMat = Transpose[# /.Rule[{rw_, cl_, id_}, val_]:>{rw-1,cl-1,val}] & /@ indexMat;
indexMat = Table[ToString[allIndex[[i]]]->indexMat[[i]], {i, 1, Length[allIndex]}];
(* \:5904\:7406\:5e38\:6570\:77e9\:9635 *)
consMat = Transpose[DeleteCases[ArrayRules[consMat], Rule[x_, 0]]/.Rule[{row_, col_}, val_]:>{row - 1, col - 1, val}];
If[consMat =!= {}, Prepend[indexMat, "Cons" -> consMat], indexMat]
]


(* ::Section::Closed:: *)
(*\:7b49\:5f0f\:7ea6\:675f\:8f6c\:5316*)


(* ::Text:: *)
(*\:5c06\:7b49\:5f0f l.h.s == r.h.s \:8f6c\:5316\:4e3a {l.h.s, r.h.s} \:7684\:5f62\:5f0f\:ff0c\:7136\:540e l/r.h.s \:8868\:793a\:4e3a {{i1, j1}, ..., {in, jn}} \:ff0c\:5373 i1 * w[j1] + ... \:7684\:5f62\:5f0f*)


PhaseEq[eq_,varmap_]:=Module[{cons,tmp,vars,coef,coeff},
tmp = eq/.varmap/.Equal->List;
tmp = tmp[[1]] - tmp[[2]];
(* \:5e38\:6570\:90e8\:5206\:63d0\:53d6
	\:5e38\:6570\:90e8\:5206\:4e5f\:53ef\:80fd\:4f1a\:542b\:6709 g \:7684\:7ebf\:6027\:9879\:ff0c\:5982 1 + 2g\:ff0c\:8fd9\:91cc\:5c06\:5176\:8868\:793a\:4e3a {1, 2} \:7684\:5f62\:5f0f
*)
cons = tmp/.w[x_]:>0;
cons = {cons/.g->0, Coefficient[cons, g]};
(* \:63d0\:53d6\:6240\:6709 w[i] \:7684\:7cfb\:6570 *)
vars = Cases[Variables[tmp], w[x_]];
coef = Coefficient[tmp, vars];
(* \:63d0\:53d6\:51fa\:6765\:7684\:7cfb\:6570 coef \:53ef\:80fd\:4f1a\:542b\:6709 g\:ff0c\:5982 (3/2g + 1)w[1] \:63d0\:53d6\:51fa\:6765\:7684\:4fbf\:662f 3/2g + 1
   \:6b64\:5904\:8fdb\:4e00\:6b65\:5c06\:5176\:63d0\:53d6\:4e3a {1, 3 / 2} \:7684\:5f62\:5f0f
*)
coeff=Table[{coef[[i]]/.g->0, Coefficient[coef[[i]],g]},{i, 1, Length[coef]}];
vars = vars/.w[x_]:>x;
Prepend[Table[{coeff[[i]],vars[[i]]},{i,1,Length[vars]}], cons]
]


(* ::Section::Closed:: *)
(*\:751f\:6210*)


(* ::Text:: *)
(*\:751f\:6210 Task.json \:6587\:4ef6*)


(* ::Text:: *)
(*target                  : \:6700\:5c0f\:5316\:7684\:76ee\:6807\:51fd\:6570 c^T . vars\:3002\:6b64\:5904\:8f93\:5165\:7684\:662f\:4e00\:4e2a List\:ff0c\:957f\:5ea6\:4e0e allVars \:76f8\:540c\:ff0c\:8868\:660e\:6bcf\:4e2a\:53d8\:91cf\:524d\:7684\:7cfb\:6570*)
(*allVars                 : \:95ee\:9898\:4e2d\:6d89\:53ca\:5230\:7684\:6240\:6709\:53d8\:91cf\:ff0c\:4e0d\:5305\:62ec\:53c2\:6570*)
(*sdpMatrix          : \:8bbe\:7f6e\:4e3a\:534a\:6b63\:5b9a\:77e9\:9635\:7ea6\:675f\:7684\:6240\:6709\:77e9\:9635*)
(*loopEquations : \:7b49\:5f0f\:7ea6\:675f*)
(*para                      : \:95ee\:9898\:4e2d\:6d89\:53ca\:5230\:7684\:53c2\:6570\:ff0c\:5b83\:5c06\:4f1a\:88ab\:5728\:540e\:9762\:7684\:7a0b\:5e8f\:7edf\:4e00\:66ff\:6362\:4e3a g*)
(*lambda               : \:53c2\:6570\:7684\:79bb\:6563\:503c*)
(*eps                        : \:6536\:655b\:7cbe\:5ea6\:ff0c\:4ec5\:652f\:6301 10^-3-10^-9*)
(*name                   : \:4efb\:52a1\:540d *)


ClearAll[GenerateTask]
GenerateTask[target_:List, allVars_ : List,sdpMatrix_ : List, loopEquations_ : List,para_, lambda_ : List, eps_:10^-6, name_:"Task"]:=Module[{vmap,allLE, dims, mats, tar, blockSize},
(* \:5224\:65ad eps \:5927\:5c0f *)
If[eps > 10^-3||eps < 10^-9,Print["EPS must be smaller than \!\(\*SuperscriptBox[\(10\), \(-3\)]\) and bigger than \!\(\*SuperscriptBox[\(10\), \(-9\)]\)"];Return];
(* \:5224\:65ad allVars \:548c target \:7684\:5f62\:72b6 *)
If[Length[target]!=Length[allVars],Print["The length of target is not same as allVars."];Return];

vmap = CVXVarMap[allVars, para];
dims = Length /@ sdpMatrix;
blockSize = Length[sdpMatrix];
Print["[Matrix] Phasing Matrices."];
mats = Monitor[Table[SuperMatrixUpper[sdpMatrix[[i]], vmap, para],{i, 1, blockSize}], ProgressIndicator[i, {1, blockSize}]];
Print["[LoopEqs] Phasing LoopEqs."];
allLE = PhaseEq[#, vmap] &/@ loopEquations;

tar = Cases[ArrayRules[target], Rule[{x_},y_]/;NumericQ[x]:>{x-1,y}];

Print["Exporting..."];
Export[name <> ".json",
{
"taskname" -> name,
"variable_length" -> Count[Normal[vmap]/.Rule[x_,y_]:>y,w[i_]],
"eps"->eps,
"para_value"->lambda,
"target"->tar,
"constrains_length"->Length[sdpMatrix],
"constrains_dim" ->dims,
"constrains"->mats,
"eqConstrains"->allLE
}, "JSON", "Compact" -> True
];
Print["The " <> name <>  ".json has been created."];
Print["Variables in allVars have been renamed. "];
(*Grid[Transpose[vmap[[;;-2]]/.Rule->List]/.w[x_]:>x]*)
]


End[];
EndPackage[];
