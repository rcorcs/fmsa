# Function Merging by Sequence Alignment

This file implements the general function merging optimization.
  
It identifies similarities between functions, and If profitable, merges them
into a single function, replacing the original ones. Functions do not need
to be identical to be merged. In fact, there is very little restriction to
merge two function, however, the produced merged function can be larger than
the two original functions together. For that reason, it uses the
TargetTransformInfo analysis to estimate the code-size costs of instructions
in order to estimate the profitability of merging two functions.

This function merging transformation has three major parts:
1. The input functions are linearized, representing their CFGs as sequences
    of labels and instructions.
2. We apply a sequence alignment algorithm, namely, the Needleman-Wunsch
    algorithm, to identify similar code between the two linearized functions.
3. We use the aligned sequences to perform code generate, producing the new
    merged function, using an extra parameter to represent the function
    identifier.

This pass integrates the function merging transformation with an exploration
framework. For every function, the other functions are ranked based their
degree of similarity, which is computed from the functions' fingerprints.
Only the top candidates are analyzed in a greedy manner and if one of them
produces a profitable result, the merged function is taken.


# Reference

"Function Merging by Sequence Alignment",
Rodrigo C. O. Rocha, Pavlos Petoumenos, Zheng Wang, Murray Cole, Hugh Leather. International Symposium on Code Generation and Optimization, 2019 (CGO'19).
