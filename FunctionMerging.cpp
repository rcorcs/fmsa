//===- FunctionMerging.cpp - A function merging pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the general function merging optimization.
//  
// It identifies similarities between functions, and If profitable, merges them
// into a single function, replacing the original ones. Functions do not need
// to be identical to be merged. In fact, there is very little restriction to
// merge two function, however, the produced merged function can be larger than
// the two original functions together. For that reason, it uses the
// TargetTransformInfo analysis to estimate the code-size costs of instructions
// in order to estimate the profitability of merging two functions.
//
// This function merging transformation has three major parts:
// 1. The input functions are linearized, representing their CFGs as sequences
//    of labels and instructions.
// 2. We apply a sequence alignment algorithm, namely, the Needleman-Wunsch
//    algorithm, to identify similar code between the two linearized functions.
// 3. We use the aligned sequences to perform code generate, producing the new
//    merged function, using an extra parameter to represent the function
//    identifier.
//
// This pass integrates the function merging transformation with an exploration
// framework. For every function, the other functions are ranked based their
// degree of similarity, which is computed from the functions' fingerprints.
// Only the top candidates are analyzed in a greedy manner and if one of them
// produces a profitable result, the merged function is taken.
// 
//===----------------------------------------------------------------------===//
//
// This optimization was proposed in
//
// Function Merging by Sequence Alignment
// Rodrigo C. O. Rocha, Pavlos Petoumenos, Zheng Wang, Murray Cole, Hugh Leather
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/FunctionMerging.h"

#include "llvm/ADT/SequenceAlignment.h"


#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include <llvm/IR/IRBuilder.h>

#include "llvm/Support/Error.h"
#include "llvm/Support/Timer.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Analysis/LoopInfo.h"
//#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/PostDominators.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

#include "llvm/Support/RandomNumberGenerator.h"

//#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Analysis/Utils/Local.h"
#include "llvm/Transforms/Utils/Local.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Utils/FunctionComparator.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#include "llvm/Transforms/IPO.h"

#include <algorithm>
#include <list>

#include <limits.h>

#include <functional>
#include <queue>
#include <vector>

#include <algorithm>
#include <stdlib.h>
#include <time.h>

#define DEBUG_TYPE "MyFuncMerge"
//#define ENABLE_DEBUG_CODE

//#define FMSA_USE_JACCARD

//#define TIME_STEPS_DEBUG

using namespace llvm;

static cl::opt<unsigned> ExplorationThreshold(
    "func-merging-explore", cl::init(10), cl::Hidden,
    cl::desc("Exploration threshold of evaluated functions"));

static cl::opt<int> MergingOverheadThreshold(
    "func-merging-threshold", cl::init(0), cl::Hidden,
    cl::desc("Threshold of allowed overhead for merging function"));

static cl::opt<bool>
    MaxParamScore("func-merging-max-param", cl::init(true), cl::Hidden,
                  cl::desc("Maximizing the score for merging parameters"));

static cl::opt<bool> Debug("func-merging-debug", cl::init(true), cl::Hidden,
                           cl::desc("Outputs debug information"));

static cl::opt<bool> Verbose("func-merging-verbose", cl::init(false),
                             cl::Hidden, cl::desc("Outputs debug information"));

static cl::opt<bool>
    IdenticalType("func-merging-identic-type", cl::init(true), cl::Hidden,
                  cl::desc("Maximizing the score for merging parameters"));

static cl::opt<bool> ApplySimilarityHeuristic(
    "func-merging-similarity-pruning", cl::init(true), cl::Hidden,
    cl::desc("Maximizing the score for merging parameters"));


static cl::opt<bool>
    EnableUnifiedReturnType("func-merging-unify-return", cl::init(true), cl::Hidden,
                  cl::desc("Enable unified return types"));


static cl::opt<bool>
    HasWholeProgram("func-merging-whole-program", cl::init(true), cl::Hidden,
                  cl::desc("Function merging applied on whole program"));


static cl::opt<bool>
    HandlePHINodes("func-merging-phi-nodes", cl::init(false), cl::Hidden,
                  cl::desc("Enable the experimental code for handling PHI nodes"));

static std::string GetValueName(const Value *V);

//TODO: make these two functions public from the original -mem2reg and -reg2mem
static void demoteRegToMem(Function &F);

static bool promoteMemoryToRegister(Function &F, DominatorTree &DT);


static bool fixNotDominatedUses(Function *F, BasicBlock *Entry, DominatorTree &DT);

FunctionMergeResult MergeFunctions(Function *F1, Function *F2,
 const FunctionMergingOptions &Options) {
  if (F1->getParent()!=F2->getParent()) return FunctionMergeResult(F1,F2,nullptr);
  FunctionMerger Merger(F1->getParent());
  return Merger.merge(F1,F2,Options);
}

// Any two pointers in the same address space are equivalent, intptr_t and
// pointers are equivalent. Otherwise, standard type equivalence rules apply.
bool FunctionMerger::areTypesEquivalent(Type *Ty1, Type *Ty2, Type *IntPtrTy, const FunctionMergingOptions &Options) {
  if (Ty1 == Ty2)
    return true;
  if (Options.IdenticalTypesOnly)
    return false;

  if (Ty1->getTypeID() != Ty2->getTypeID()) {
    if (isa<PointerType>(Ty1) && Ty2 == IntPtrTy)
      return true;
    if (isa<PointerType>(Ty2) && Ty1 == IntPtrTy)
      return true;
    return false;
  }

  switch (Ty1->getTypeID()) {
  default:
    llvm_unreachable("Unknown type!");
    // Fall through in Release mode.
  case Type::IntegerTyID:
  case Type::VectorTyID:
    // Ty1 == Ty2 would have returned true earlier.
    return false;
  case Type::VoidTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::LabelTyID:
  case Type::MetadataTyID:
    return true;
  case Type::PointerTyID: {
    //pointers should always be valid
    //we just need to cast them
    PointerType *PTy1 = cast<PointerType>(Ty1);
    PointerType *PTy2 = cast<PointerType>(Ty2);
    return (PTy1 == PTy2);
  }
  case Type::StructTyID: {
    StructType *STy1 = cast<StructType>(Ty1);
    StructType *STy2 = cast<StructType>(Ty2);
    if (STy1->getNumElements() != STy2->getNumElements())
      return false;

    if (STy1->isPacked() != STy2->isPacked())
      return false;

    for (unsigned i = 0, e = STy1->getNumElements(); i != e; ++i) {
      if (!areTypesEquivalent(STy1->getElementType(i), STy2->getElementType(i),
                            IntPtrTy, Options))
        return false;
    }
    return true;
  }
  case Type::FunctionTyID: {
    FunctionType *FTy1 = cast<FunctionType>(Ty1);
    FunctionType *FTy2 = cast<FunctionType>(Ty2);
    if (FTy1->getNumParams() != FTy2->getNumParams() ||
        FTy1->isVarArg() != FTy2->isVarArg())
      return false;

    if (!areTypesEquivalent(FTy1->getReturnType(), FTy2->getReturnType(), IntPtrTy, Options))
      return false;

    for (unsigned i = 0, e = FTy1->getNumParams(); i != e; ++i) {
      if (!areTypesEquivalent(FTy1->getParamType(i), FTy2->getParamType(i), IntPtrTy, Options))
        return false;
    }
    return true;
  }
  case Type::ArrayTyID: {
    ArrayType *ATy1 = cast<ArrayType>(Ty1);
    ArrayType *ATy2 = cast<ArrayType>(Ty2);
    return ATy1->getNumElements() == ATy2->getNumElements() &&
           areTypesEquivalent(ATy1->getElementType(), ATy2->getElementType(), IntPtrTy, Options);
  }
  }
  return false;
}

// Any two pointers in the same address space are equivalent, intptr_t and
// pointers are equivalent. Otherwise, standard type equivalence rules apply.
bool FunctionMerger::areTypesEquivalent(Type *Ty1, Type *Ty2, const FunctionMergingOptions &Options) {
 return areTypesEquivalent(Ty1, Ty2, IntPtrTy, Options);
}

bool FunctionMerger::matchIntrinsicCalls(Intrinsic::ID ID, const CallInst *CI1,
                                const CallInst *CI2) {
  Intrinsic::ID ID1;
  Intrinsic::ID ID2;
  if (Function *F = CI1->getCalledFunction())
    ID1 = (Intrinsic::ID)F->getIntrinsicID();
  if (Function *F = CI2->getCalledFunction())
    ID2 = (Intrinsic::ID)F->getIntrinsicID();

  if (ID1 != ID)
    return false;
  if (ID1 != ID2)
    return false;

  switch (ID) {
  default:
    break;
  case Intrinsic::coro_id: {
    /*
    auto *InfoArg = CS.getArgOperand(3)->stripPointerCasts();
    if (isa<ConstantPointerNull>(InfoArg))
      break;
    auto *GV = dyn_cast<GlobalVariable>(InfoArg);
    Assert(GV && GV->isConstant() && GV->hasDefinitiveInitializer(),
      "info argument of llvm.coro.begin must refer to an initialized "
      "constant");
    Constant *Init = GV->getInitializer();
    Assert(isa<ConstantStruct>(Init) || isa<ConstantArray>(Init),
      "info argument of llvm.coro.begin must refer to either a struct or "
      "an array");
    */
    break;
  }
  case Intrinsic::ctlz: // llvm.ctlz
  case Intrinsic::cttz: // llvm.cttz
    //is_zero_undef argument of bit counting intrinsics must be a constant int
    return CI1->getArgOperand(1) == CI2->getArgOperand(1);
  case Intrinsic::experimental_constrained_fadd:
  case Intrinsic::experimental_constrained_fsub:
  case Intrinsic::experimental_constrained_fmul:
  case Intrinsic::experimental_constrained_fdiv:
  case Intrinsic::experimental_constrained_frem:
  case Intrinsic::experimental_constrained_fma:
  case Intrinsic::experimental_constrained_sqrt:
  case Intrinsic::experimental_constrained_pow:
  case Intrinsic::experimental_constrained_powi:
  case Intrinsic::experimental_constrained_sin:
  case Intrinsic::experimental_constrained_cos:
  case Intrinsic::experimental_constrained_exp:
  case Intrinsic::experimental_constrained_exp2:
  case Intrinsic::experimental_constrained_log:
  case Intrinsic::experimental_constrained_log10:
  case Intrinsic::experimental_constrained_log2:
  case Intrinsic::experimental_constrained_rint:
  case Intrinsic::experimental_constrained_nearbyint:
    // visitConstrainedFPIntrinsic(
    //    cast<ConstrainedFPIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_declare: // llvm.dbg.declare
    // Assert(isa<MetadataAsValue>(CS.getArgOperand(0)),
    //       "invalid llvm.dbg.declare intrinsic call 1", CS);
    // visitDbgIntrinsic("declare",
    // cast<DbgInfoIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_addr: // llvm.dbg.addr
    // visitDbgIntrinsic("addr", cast<DbgInfoIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_value: // llvm.dbg.value
    // visitDbgIntrinsic("value", cast<DbgInfoIntrinsic>(*CS.getInstruction()));
    break;
  case Intrinsic::dbg_label: // llvm.dbg.label
    // visitDbgLabelIntrinsic("label",
    // cast<DbgLabelInst>(*CS.getInstruction()));
    break;
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset: {
    //isvolatile argument of memory intrinsics must be a constant int
    return CI1->getArgOperand(3) == CI2->getArgOperand(3);
  }
  case Intrinsic::memcpy_element_unordered_atomic:
  case Intrinsic::memmove_element_unordered_atomic:
  case Intrinsic::memset_element_unordered_atomic: {
    const auto *AMI1 = cast<AtomicMemIntrinsic>(CI1);
    const auto *AMI2 = cast<AtomicMemIntrinsic>(CI2);

    ConstantInt *ElementSizeCI1 =
        dyn_cast<ConstantInt>(AMI1->getRawElementSizeInBytes());

    ConstantInt *ElementSizeCI2 =
        dyn_cast<ConstantInt>(AMI2->getRawElementSizeInBytes());

    return (ElementSizeCI1!=nullptr && ElementSizeCI1==ElementSizeCI2);
  }
  case Intrinsic::gcroot:
  case Intrinsic::gcwrite:
  case Intrinsic::gcread:
    //llvm.gcroot parameter #2 must be a constant.
    return CI1->getArgOperand(1) == CI2->getArgOperand(1);
  case Intrinsic::init_trampoline:
    break;
  case Intrinsic::prefetch:
    //arguments #2 and #3 in llvm.prefetch must be constants
    return CI1->getArgOperand(1) == CI2->getArgOperand(1) &&
           CI1->getArgOperand(2) == CI2->getArgOperand(2);
  case Intrinsic::stackprotector:
    /*
    Assert(isa<AllocaInst>(CS.getArgOperand(1)->stripPointerCasts()),
           "llvm.stackprotector parameter #2 must resolve to an alloca.", CS);
    */
    break;
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
    //size argument of memory use markers must be a constant integer
    return CI1->getArgOperand(0) == CI2->getArgOperand(0);
  case Intrinsic::invariant_end:
    //llvm.invariant.end parameter #2 must be a constant integer
    return CI1->getArgOperand(1) == CI2->getArgOperand(1);
  case Intrinsic::localescape: {
    /*
    BasicBlock *BB = CS.getParent();
    Assert(BB == &BB->getParent()->front(),
           "llvm.localescape used outside of entry block", CS);
    Assert(!SawFrameEscape,
           "multiple calls to llvm.localescape in one function", CS);
    for (Value *Arg : CS.args()) {
      if (isa<ConstantPointerNull>(Arg))
        continue; // Null values are allowed as placeholders.
      auto *AI = dyn_cast<AllocaInst>(Arg->stripPointerCasts());
      Assert(AI && AI->isStaticAlloca(),
             "llvm.localescape only accepts static allocas", CS);
    }
    FrameEscapeInfo[BB->getParent()].first = CS.getNumArgOperands();
    SawFrameEscape = true;
    */
    break;
  }
  case Intrinsic::localrecover: {
    /*
    Value *FnArg = CS.getArgOperand(0)->stripPointerCasts();
    Function *Fn = dyn_cast<Function>(FnArg);
    Assert(Fn && !Fn->isDeclaration(),
           "llvm.localrecover first "
           "argument must be function defined in this module",
           CS);
    auto *IdxArg = dyn_cast<ConstantInt>(CS.getArgOperand(2));
    Assert(IdxArg, "idx argument of llvm.localrecover must be a constant int",
           CS);
    auto &Entry = FrameEscapeInfo[Fn];
    Entry.second = unsigned(
        std::max(uint64_t(Entry.second), IdxArg->getLimitedValue(~0U) + 1));
    */
    break;
  }
    /*
    case Intrinsic::experimental_gc_statepoint:
      Assert(!CS.isInlineAsm(),
             "gc.statepoint support for inline assembly unimplemented", CS);
      Assert(CS.getParent()->getParent()->hasGC(),
             "Enclosing function does not use GC.", CS);

      verifyStatepoint(CS);
      break;
    case Intrinsic::experimental_gc_result: {
      Assert(CS.getParent()->getParent()->hasGC(),
             "Enclosing function does not use GC.", CS);
      // Are we tied to a statepoint properly?
      CallSite StatepointCS(CS.getArgOperand(0));
      const Function *StatepointFn =
        StatepointCS.getInstruction() ? StatepointCS.getCalledFunction() :
    nullptr; Assert(StatepointFn && StatepointFn->isDeclaration() &&
                 StatepointFn->getIntrinsicID() ==
                     Intrinsic::experimental_gc_statepoint,
             "gc.result operand #1 must be from a statepoint", CS,
             CS.getArgOperand(0));

      // Assert that result type matches wrapped callee.
      const Value *Target = StatepointCS.getArgument(2);
      auto *PT = cast<PointerType>(Target->getType());
      auto *TargetFuncType = cast<FunctionType>(PT->getElementType());
      Assert(CS.getType() == TargetFuncType->getReturnType(),
             "gc.result result type does not match wrapped callee", CS);
      break;
    }
    case Intrinsic::experimental_gc_relocate: {
      Assert(CS.getNumArgOperands() == 3, "wrong number of arguments", CS);

      Assert(isa<PointerType>(CS.getType()->getScalarType()),
             "gc.relocate must return a pointer or a vector of pointers", CS);

      // Check that this relocate is correctly tied to the statepoint

      // This is case for relocate on the unwinding path of an invoke statepoint
      if (LandingPadInst *LandingPad =
            dyn_cast<LandingPadInst>(CS.getArgOperand(0))) {

        const BasicBlock *InvokeBB =
            LandingPad->getParent()->getUniquePredecessor();

        // Landingpad relocates should have only one predecessor with invoke
        // statepoint terminator
        Assert(InvokeBB, "safepoints should have unique landingpads",
               LandingPad->getParent());
        Assert(InvokeBB->getTerminator(), "safepoint block should be well
    formed", InvokeBB); Assert(isStatepoint(InvokeBB->getTerminator()), "gc
    relocate should be linked to a statepoint", InvokeBB);
      }
      else {
        // In all other cases relocate should be tied to the statepoint
    directly.
        // This covers relocates on a normal return path of invoke statepoint
    and
        // relocates of a call statepoint.
        auto Token = CS.getArgOperand(0);
        Assert(isa<Instruction>(Token) &&
    isStatepoint(cast<Instruction>(Token)), "gc relocate is incorrectly tied to
    the statepoint", CS, Token);
      }

      // Verify rest of the relocate arguments.

      ImmutableCallSite StatepointCS(
          cast<GCRelocateInst>(*CS.getInstruction()).getStatepoint());

      // Both the base and derived must be piped through the safepoint.
      Value* Base = CS.getArgOperand(1);
      Assert(isa<ConstantInt>(Base),
             "gc.relocate operand #2 must be integer offset", CS);

      Value* Derived = CS.getArgOperand(2);
      Assert(isa<ConstantInt>(Derived),
             "gc.relocate operand #3 must be integer offset", CS);

      const int BaseIndex = cast<ConstantInt>(Base)->getZExtValue();
      const int DerivedIndex = cast<ConstantInt>(Derived)->getZExtValue();
      // Check the bounds
      Assert(0 <= BaseIndex && BaseIndex < (int)StatepointCS.arg_size(),
             "gc.relocate: statepoint base index out of bounds", CS);
      Assert(0 <= DerivedIndex && DerivedIndex < (int)StatepointCS.arg_size(),
             "gc.relocate: statepoint derived index out of bounds", CS);

      // Check that BaseIndex and DerivedIndex fall within the 'gc parameters'
      // section of the statepoint's argument.
      Assert(StatepointCS.arg_size() > 0,
             "gc.statepoint: insufficient arguments");
      Assert(isa<ConstantInt>(StatepointCS.getArgument(3)),
             "gc.statement: number of call arguments must be constant integer");
      const unsigned NumCallArgs =
          cast<ConstantInt>(StatepointCS.getArgument(3))->getZExtValue();
      Assert(StatepointCS.arg_size() > NumCallArgs + 5,
             "gc.statepoint: mismatch in number of call arguments");
      Assert(isa<ConstantInt>(StatepointCS.getArgument(NumCallArgs + 5)),
             "gc.statepoint: number of transition arguments must be "
             "a constant integer");
      const int NumTransitionArgs =
          cast<ConstantInt>(StatepointCS.getArgument(NumCallArgs + 5))
              ->getZExtValue();
      const int DeoptArgsStart = 4 + NumCallArgs + 1 + NumTransitionArgs + 1;
      Assert(isa<ConstantInt>(StatepointCS.getArgument(DeoptArgsStart)),
             "gc.statepoint: number of deoptimization arguments must be "
             "a constant integer");
      const int NumDeoptArgs =
          cast<ConstantInt>(StatepointCS.getArgument(DeoptArgsStart))
              ->getZExtValue();
      const int GCParamArgsStart = DeoptArgsStart + 1 + NumDeoptArgs;
      const int GCParamArgsEnd = StatepointCS.arg_size();
      Assert(GCParamArgsStart <= BaseIndex && BaseIndex < GCParamArgsEnd,
             "gc.relocate: statepoint base index doesn't fall within the "
             "'gc parameters' section of the statepoint call",
             CS);
      Assert(GCParamArgsStart <= DerivedIndex && DerivedIndex < GCParamArgsEnd,
             "gc.relocate: statepoint derived index doesn't fall within the "
             "'gc parameters' section of the statepoint call",
             CS);

      // Relocated value must be either a pointer type or vector-of-pointer
    type,
      // but gc_relocate does not need to return the same pointer type as the
      // relocated pointer. It can be casted to the correct type later if it's
      // desired. However, they must have the same address space and
    'vectorness' GCRelocateInst &Relocate =
    cast<GCRelocateInst>(*CS.getInstruction());
      Assert(Relocate.getDerivedPtr()->getType()->isPtrOrPtrVectorTy(),
             "gc.relocate: relocated value must be a gc pointer", CS);

      auto ResultType = CS.getType();
      auto DerivedType = Relocate.getDerivedPtr()->getType();
      Assert(ResultType->isVectorTy() == DerivedType->isVectorTy(),
             "gc.relocate: vector relocates to vector and pointer to pointer",
             CS);
      Assert(
          ResultType->getPointerAddressSpace() ==
              DerivedType->getPointerAddressSpace(),
          "gc.relocate: relocating a pointer shouldn't change its address
    space", CS); break;
    }
    case Intrinsic::eh_exceptioncode:
    case Intrinsic::eh_exceptionpointer: {
      Assert(isa<CatchPadInst>(CS.getArgOperand(0)),
             "eh.exceptionpointer argument must be a catchpad", CS);
      break;
    }
    case Intrinsic::masked_load: {
      Assert(CS.getType()->isVectorTy(), "masked_load: must return a vector",
    CS);

      Value *Ptr = CS.getArgOperand(0);
      //Value *Alignment = CS.getArgOperand(1);
      Value *Mask = CS.getArgOperand(2);
      Value *PassThru = CS.getArgOperand(3);
      Assert(Mask->getType()->isVectorTy(),
             "masked_load: mask must be vector", CS);

      // DataTy is the overloaded type
      Type *DataTy = cast<PointerType>(Ptr->getType())->getElementType();
      Assert(DataTy == CS.getType(),
             "masked_load: return must match pointer type", CS);
      Assert(PassThru->getType() == DataTy,
             "masked_load: pass through and data type must match", CS);
      Assert(Mask->getType()->getVectorNumElements() ==
             DataTy->getVectorNumElements(),
             "masked_load: vector mask must be same length as data", CS);
      break;
    }
    case Intrinsic::masked_store: {
      Value *Val = CS.getArgOperand(0);
      Value *Ptr = CS.getArgOperand(1);
      //Value *Alignment = CS.getArgOperand(2);
      Value *Mask = CS.getArgOperand(3);
      Assert(Mask->getType()->isVectorTy(),
             "masked_store: mask must be vector", CS);

      // DataTy is the overloaded type
      Type *DataTy = cast<PointerType>(Ptr->getType())->getElementType();
      Assert(DataTy == Val->getType(),
             "masked_store: storee must match pointer type", CS);
      Assert(Mask->getType()->getVectorNumElements() ==
             DataTy->getVectorNumElements(),
             "masked_store: vector mask must be same length as data", CS);
      break;
    }

    case Intrinsic::experimental_guard: {
      Assert(CS.isCall(), "experimental_guard cannot be invoked", CS);
      Assert(CS.countOperandBundlesOfType(LLVMContext::OB_deopt) == 1,
             "experimental_guard must have exactly one "
             "\"deopt\" operand bundle");
      break;
    }

    case Intrinsic::experimental_deoptimize: {
      Assert(CS.isCall(), "experimental_deoptimize cannot be invoked", CS);
      Assert(CS.countOperandBundlesOfType(LLVMContext::OB_deopt) == 1,
             "experimental_deoptimize must have exactly one "
             "\"deopt\" operand bundle");
      Assert(CS.getType() ==
    CS.getInstruction()->getFunction()->getReturnType(),
             "experimental_deoptimize return type must match caller return
    type");

      if (CS.isCall()) {
        auto *DeoptCI = CS.getInstruction();
        auto *RI = dyn_cast<ReturnInst>(DeoptCI->getNextNode());
        Assert(RI,
               "calls to experimental_deoptimize must be followed by a return");

        if (!CS.getType()->isVoidTy() && RI)
          Assert(RI->getReturnValue() == DeoptCI,
                 "calls to experimental_deoptimize must be followed by a return
    " "of the value computed by experimental_deoptimize");
      }

      break;
    }
    */
  };
  return false; // TODO: change to false by default
}

bool FunctionMerger::matchLandingPad(LandingPadInst *LP1, LandingPadInst *LP2) {
  if (LP1->getType() != LP2->getType())
    return false;
  if (LP1->isCleanup() != LP2->isCleanup())
    return false;
  if (LP1->getNumClauses() != LP2->getNumClauses())
    return false;
  for (unsigned i = 0; i < LP1->getNumClauses(); i++) {
    if (LP1->isCatch(i) != LP2->isCatch(i))
      return false;
    if (LP1->isFilter(i) != LP2->isFilter(i))
      return false;
    if (LP1->getClause(i) != LP2->getClause(i))
      return false;
  }
  return true;
}

bool FunctionMerger::matchInstructions(Instruction *I1, Instruction *I2, const FunctionMergingOptions &Options) {

  if (I1->getOpcode() != I2->getOpcode()) return false;

  //Returns are special cases that can differ in the number of operands
  if (I1->getOpcode() == Instruction::Ret)
    return true;

  if (I1->getNumOperands() != I2->getNumOperands())
    return false;

  bool sameType = false;
  if (Options.IdenticalTypesOnly) {
    sameType = (I1->getType() == I2->getType());
    for (unsigned i = 0; i < I1->getNumOperands(); i++) {
      sameType = sameType &&
                 (I1->getOperand(i)->getType() == I2->getOperand(i)->getType());
    }
  } else {
    Module *M = ((Module *)I1->getParent()->getParent()->getParent());
    const DataLayout *DL = &M->getDataLayout();
    LLVMContext &Context = M->getContext();
    Type *IntPtrTy = DL->getIntPtrType(Context);

    sameType = areTypesEquivalent(I1->getType(), I2->getType(), IntPtrTy, Options);
    for (unsigned i = 0; i < I1->getNumOperands(); i++) {
      sameType = sameType && areTypesEquivalent(I1->getOperand(i)->getType(),
                                              I2->getOperand(i)->getType(), IntPtrTy, Options);
    }
  }
  if (!sameType)
    return false;

  switch (I1->getOpcode()) {

  case Instruction::Load: {
    const LoadInst *LI = dyn_cast<LoadInst>(I1);
    const LoadInst *LI2 = cast<LoadInst>(I2);
    return LI->isVolatile() == LI2->isVolatile() &&
           LI->getAlignment() == LI2->getAlignment() &&
           LI->getOrdering() == LI2->getOrdering(); // &&
    // LI->getSyncScopeID() == LI2->getSyncScopeID() &&
    // LI->getMetadata(LLVMContext::MD_range)
    //  == LI2->getMetadata(LLVMContext::MD_range);
  }
  case Instruction::Store: {
    const StoreInst *SI = dyn_cast<StoreInst>(I1);
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           SI->getAlignment() == cast<StoreInst>(I2)->getAlignment() &&
           SI->getOrdering() == cast<StoreInst>(I2)->getOrdering(); // &&
    // SI->getSyncScopeID() == cast<StoreInst>(I2)->getSyncScopeID();
  }
  case Instruction::Alloca: {
    const AllocaInst *AI = dyn_cast<AllocaInst>(I1);
    if (AI->getArraySize() != cast<AllocaInst>(I2)->getArraySize() ||
        AI->getAlignment() != cast<AllocaInst>(I2)->getAlignment())
      return false;

    /*
    // If size is known, I2 can be seen as equivalent to I1 if it allocates
    // the same or less memory.
    if (DL->getTypeAllocSize(AI->getAllocatedType())
          < DL->getTypeAllocSize(cast<AllocaInst>(I2)->getAllocatedType()))
      return false;

    return true;
    */
    break;
  }
  case Instruction::GetElementPtr: {
    GetElementPtrInst *GEP1 = dyn_cast<GetElementPtrInst>(I1);
    GetElementPtrInst *GEP2 = dyn_cast<GetElementPtrInst>(I2);

    SmallVector<Value *, 8> Indices1(GEP1->idx_begin(), GEP1->idx_end());
    SmallVector<Value *, 8> Indices2(GEP2->idx_begin(), GEP2->idx_end());
    if (Indices1.size() != Indices2.size())
      return false;

    /*
    //TODO: some indices must be constant depending on the type being indexed.
    //For simplicity, whenever a given index is constant, keep it constant.
    //This simplification may degrade the merging quality.
    for (unsigned i = 0; i < Indices1.size(); i++) {
      if (isa<ConstantInt>(Indices1[i]) && isa<ConstantInt>(Indices2[i]) && Indices1[i] != Indices2[i])
        return false; // if different constant values
    }
    */

    Type *AggTy1 = GEP1->getSourceElementType();
    Type *AggTy2 = GEP2->getSourceElementType();

    unsigned CurIdx = 1;
    for (; CurIdx != Indices1.size(); ++CurIdx) {
      CompositeType *CTy1 = dyn_cast<CompositeType>(AggTy1);
      CompositeType *CTy2 = dyn_cast<CompositeType>(AggTy2);
      if (!CTy1 || CTy1->isPointerTy()) return false;
      if (!CTy2 || CTy2->isPointerTy()) return false;
      Value *Idx1 = Indices1[CurIdx];
      Value *Idx2 = Indices2[CurIdx];
      //if (!CT->indexValid(Index)) return nullptr;
      
      //validate indices
      if (isa<StructType>(CTy1) || isa<StructType>(CTy2)) {
        //if types are structs, the indices must be and remain constants
        if (!isa<ConstantInt>(Idx1) || !isa<ConstantInt>(Idx2)) return false;
        if (Idx1!=Idx2) return false;
      }

      AggTy1 = CTy1->getTypeAtIndex(Idx1);
      AggTy2 = CTy2->getTypeAtIndex(Idx2);

      //sanity check: matching indexed types
      bool sameType = (AggTy1 == AggTy2);
      if (!Options.IdenticalTypesOnly) {
        Module *M = ((Module *)I1->getParent()->getParent()->getParent());
        const DataLayout *DL = &M->getDataLayout();
        LLVMContext &Context = M->getContext();
        Type *IntPtrTy = DL->getIntPtrType(Context);
        sameType = areTypesEquivalent(AggTy1, AggTy2, IntPtrTy, Options);
      }
      if (!sameType) return false;
    }
   
    break;
  }
  case Instruction::Switch: {
    SwitchInst *SI1 = dyn_cast<SwitchInst>(I1);
    SwitchInst *SI2 = dyn_cast<SwitchInst>(I2);
    if (SI1->getNumCases() == SI2->getNumCases()) {
      auto CaseIt1 = SI1->case_begin(), CaseEnd1 = SI1->case_end();
      auto CaseIt2 = SI2->case_begin(), CaseEnd2 = SI2->case_end();
      do {
        auto *Case1 = &*CaseIt1;
        auto *Case2 = &*CaseIt2;
        if (Case1 != Case2)
          return false; // TODO: could allow permutation!
        ++CaseIt1;
        ++CaseIt2;
      } while (CaseIt1 != CaseEnd1 && CaseIt2 != CaseEnd2);
      return true;
    }
    return false;
  }
  case Instruction::Call: {
    CallInst *CI1 = dyn_cast<CallInst>(I1);
    CallInst *CI2 = dyn_cast<CallInst>(I2);
    if (CI1->isInlineAsm() || CI2->isInlineAsm())
      return false;
    if (CI1->getCalledFunction() != CI2->getCalledFunction())
      return false;
    if (Function *F = CI1->getCalledFunction()) {
      if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {

        if (!matchIntrinsicCalls(ID, CI1, CI2))
          return false;
      }
    }

    return CI1->getCallingConv() ==
           CI2->getCallingConv(); // &&
                                  // CI->getAttributes() ==
                                  // cast<CallInst>(I2)->getAttributes();
  }
  case Instruction::Invoke: {
    InvokeInst *CI1 = dyn_cast<InvokeInst>(I1);
    InvokeInst *CI2 = dyn_cast<InvokeInst>(I2);
    return CI1->getCallingConv() == CI2->getCallingConv() &&
           matchLandingPad(CI1->getLandingPadInst(), CI2->getLandingPadInst());
    // CI->getAttributes() == cast<InvokeInst>(I2)->getAttributes();
  }
  case Instruction::InsertValue: {
    const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1);
    return IVI->getIndices() == cast<InsertValueInst>(I2)->getIndices();
  }
  case Instruction::ExtractValue: {
    const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1);
    return EVI->getIndices() == cast<ExtractValueInst>(I2)->getIndices();
  }
  case Instruction::Fence: {
    const FenceInst *FI = dyn_cast<FenceInst>(I1);
    return FI->getOrdering() == cast<FenceInst>(I2)->getOrdering() &&
           FI->getSyncScopeID() == cast<FenceInst>(I2)->getSyncScopeID();
  }
  case Instruction::AtomicCmpXchg: {
    const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I1);
    const AtomicCmpXchgInst *CXI2 = cast<AtomicCmpXchgInst>(I2);
    return CXI->isVolatile() == CXI2->isVolatile() &&
           CXI->isWeak() == CXI2->isWeak() &&
           CXI->getSuccessOrdering() == CXI2->getSuccessOrdering() &&
           CXI->getFailureOrdering() == CXI2->getFailureOrdering() &&
           CXI->getSyncScopeID() == CXI2->getSyncScopeID();
  }
  case Instruction::AtomicRMW: {
    const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I1);
    return RMWI->getOperation() == cast<AtomicRMWInst>(I2)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I2)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I2)->getOrdering() &&
           RMWI->getSyncScopeID() == cast<AtomicRMWInst>(I2)->getSyncScopeID();
  }
  default:
    if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
      return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  }

  return true;
}

bool FunctionMerger::match(Value *V1, Value *V2) {
  if (isa<Instruction>(V1) && isa<Instruction>(V2)) {
    return matchInstructions(dyn_cast<Instruction>(V1), dyn_cast<Instruction>(V2));
  } else if (isa<BasicBlock>(V1) && isa<BasicBlock>(V2)) {
    BasicBlock *BB1 = dyn_cast<BasicBlock>(V1);
    BasicBlock *BB2 = dyn_cast<BasicBlock>(V2);
    if (BB1->isLandingPad() || BB2->isLandingPad()) {
      LandingPadInst *LP1 = BB1->getLandingPadInst();
      LandingPadInst *LP2 = BB2->getLandingPadInst();
      if (LP1 == nullptr || LP2 == nullptr)
        return false;
      return matchLandingPad(LP1, LP2);
    } else return true;
  }
  return false;
}


static unsigned
RandomLinearizationOfBlocks(BasicBlock *BB,
                            std::list<BasicBlock *> &OrederedBBs,
                            std::set<BasicBlock *> &Visited) {
  if (Visited.find(BB) != Visited.end())
    return 0;
  Visited.insert(BB);

  TerminatorInst *TI = BB->getTerminator();

  std::vector<BasicBlock *> NextBBs;
  for (unsigned i = 0; i < TI->getNumSuccessors(); i++) {
    NextBBs.push_back(TI->getSuccessor(i));
  }
  std::random_shuffle(NextBBs.begin(), NextBBs.end());

  unsigned SumSizes = 0;
  for (BasicBlock *NextBlock : NextBBs) {
    SumSizes += RandomLinearizationOfBlocks(NextBlock, OrederedBBs, Visited);
  }

  OrederedBBs.push_front(BB);
  return SumSizes + BB->size();
}

static unsigned
RandomLinearizationOfBlocks(Function *F, std::list<BasicBlock *> &OrederedBBs) {
  std::set<BasicBlock *> Visited;
  return RandomLinearizationOfBlocks(&F->getEntryBlock(), OrederedBBs, Visited);
}

static unsigned
CanonicalLinearizationOfBlocks(BasicBlock *BB,
                               std::list<BasicBlock *> &OrederedBBs,
                               std::set<BasicBlock *> &Visited) {
  if (Visited.find(BB) != Visited.end())
    return 0;
  Visited.insert(BB);

  TerminatorInst *TI = BB->getTerminator();

  unsigned SumSizes = 0;
  for (unsigned i = 0; i < TI->getNumSuccessors(); i++) {
    SumSizes += CanonicalLinearizationOfBlocks(TI->getSuccessor(i), OrederedBBs,
                                               Visited);
  }

  OrederedBBs.push_front(BB);
  return SumSizes + BB->size();
}

static unsigned
CanonicalLinearizationOfBlocks(Function *F,
                               std::list<BasicBlock *> &OrederedBBs) {
  std::set<BasicBlock *> Visited;
  return CanonicalLinearizationOfBlocks(&F->getEntryBlock(), OrederedBBs,
                                        Visited);
}

void FunctionMerger::linearize(Function *F, SmallVectorImpl<Value *> &FVec,
                          FunctionMerger::LinearizationKind LK) {
  std::list<BasicBlock *> OrderedBBs;

  unsigned FReserve = 0;
  switch (LK) {
  case LinearizationKind::LK_Random:
    FReserve = RandomLinearizationOfBlocks(F, OrderedBBs);
  case LinearizationKind::LK_Canonical:
  default:
    FReserve = CanonicalLinearizationOfBlocks(F, OrderedBBs);
  }

  FVec.reserve(FReserve + OrderedBBs.size());
  for (BasicBlock *BB : OrderedBBs) {
    FVec.push_back(BB);
    for (Instruction &I : *BB) {
      if (!isa<LandingPadInst>(&I) && !isa<PHINode>(&I)) {
        FVec.push_back(&I);
      }
    }
  }
}

bool FunctionMerger::validMergeTypes(Function *F1, Function *F2, const FunctionMergingOptions &Options) {
  bool EquivTypes = areTypesEquivalent(F1->getReturnType(), F2->getReturnType(),
                                       Options);
  if (!EquivTypes) {
    if (!F1->getReturnType()->isVoidTy() && !F2->getReturnType()->isVoidTy())
      return false;
  }
  return true;
}

struct SelectCacheEntry {
public:
  Value *Cond;
  Value *ValTrue;
  Value *ValFalse;
  BasicBlock *Block;

  SelectCacheEntry(Value *C, Value *V1, Value *V2, BasicBlock *BB)
      : Cond(C), ValTrue(V1), ValFalse(V2), Block(BB) {}

  bool operator<(const SelectCacheEntry &Other) const {
    if (Cond != Other.Cond)
      return Cond < Other.Cond;
    if (ValTrue != Other.ValTrue)
      return ValTrue < Other.ValTrue;
    if (ValFalse != Other.ValFalse)
      return ValFalse < Other.ValFalse;
    if (Block != Other.Block)
      return Block < Other.Block;
    return false;
  }
};


#ifdef TIME_STEPS_DEBUG
Timer TimeAlign("Merge::Align", "Merge::Align");
Timer TimeParam("Merge::Param", "Merge::Param");
Timer TimeCodeGen1("Merge::CodeGen1", "Merge::CodeGen1");
Timer TimeCodeGen2("Merge::CodeGen2", "Merge::CodeGen2");
Timer TimeCodeGenFix("Merge::CodeGenFix", "Merge::CodeGenFix");
#endif


static bool validMergePair(Function *F1, Function *F2) {
  if (!F1->getSection().equals(F2->getSection())) return false;

  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 != PersonalityFn2) return false;
  }

  return true;
}

static void MergeArguments(LLVMContext &Context, Function *F1, Function *F2, std::list<std::pair<Value *, Value *>> &AlignedInsts, std::map<unsigned, unsigned> &ParamMap1, std::map<unsigned, unsigned> &ParamMap2, std::vector<Type *> &Args, const FunctionMergingOptions &Options) {

  std::vector<Argument *> ArgsList1;
  for (Argument &arg : F1->args()) {
    ArgsList1.push_back(&arg);
  }

  Args.push_back(IntegerType::get(Context, 1)); // push the function Id argument
  unsigned ArgId = 0;
  for (auto I = F1->arg_begin(), E = F1->arg_end(); I != E; I++) {
    ParamMap1[ArgId] = Args.size();
    Args.push_back((*I).getType());
    ArgId++;
  }

  // merge arguments from Function2 with Function1
  ArgId = 0;
  for (auto I = F2->arg_begin(), E = F2->arg_end(); I != E; I++) {

    std::map<unsigned, int> MatchingScore;
    // first try to find an argument with the same name/type
    // otherwise try to match by type only
    for (unsigned i = 0; i < ArgsList1.size(); i++) {
      if (ArgsList1[i]->getType() == (*I).getType()) {
        bool hasConflict = false; // check for conflict from a previous matching
        for (auto ParamPair : ParamMap2) {
          if (ParamPair.second == ParamMap1[i]) {
            hasConflict = true;
            break;
          }
        }
        if (hasConflict)
          continue;
        MatchingScore[i] = 0;
        if (!Options.MaximizeParamScore)
          break; // if not maximize score, get the first one
      }
    }

    if (MatchingScore.size() > 0) { // maximize scores
      for (auto Pair : AlignedInsts) {
        if (Pair.first != nullptr && Pair.second != nullptr) {
          auto *I1 = dyn_cast<Instruction>(Pair.first);
          auto *I2 = dyn_cast<Instruction>(Pair.second);
          if (I1 != nullptr && I2 != nullptr) { // test both for sanity
            for (unsigned i = 0; i < I1->getNumOperands(); i++) {
              for (auto KV : MatchingScore) {
                if (I1->getOperand(i) == ArgsList1[KV.first]) {
                  if (i < I2->getNumOperands() && I2->getOperand(i) == &(*I)) {
                    MatchingScore[KV.first]++;
                  }
                }
              }
            }
          }
        }
      }

      int MaxScore = -1;
      int MaxId = 0;

      for (auto KV : MatchingScore) {
        if (KV.second > MaxScore) {
          MaxScore = KV.second;
          MaxId = KV.first;
        }
      }

      ParamMap2[ArgId] = ParamMap1[MaxId];
    } else {
      ParamMap2[ArgId] = Args.size();
      Args.push_back((*I).getType());
    }

    ArgId++;
  }

}

static void SetFunctionAttributes(Function *F1, Function *F2, Function *MergedFunc) {
  unsigned MaxAlignment = std::max(F1->getAlignment(), F2->getAlignment());
  MergedFunc->setAlignment(MaxAlignment);

  if (F1->getCallingConv() == F2->getCallingConv()) {
    MergedFunc->setCallingConv(F1->getCallingConv());
  } else {
    errs() << "ERROR: different calling convention!\n";
    MergedFunc->setCallingConv(CallingConv::Fast);
  }

  if (F1->getLinkage() == F2->getLinkage()) {
    MergedFunc->setLinkage(F1->getLinkage());
  } else {
    errs() << "ERROR: different linkage type!\n";
    MergedFunc->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
  }

  if (F1->isDSOLocal() == F2->isDSOLocal()) {
    MergedFunc->setDSOLocal(F1->isDSOLocal());
  } else {
    errs() << "ERROR: different DSO local!\n";
    //MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
  }


  if (F1->getSubprogram() == F2->getSubprogram()) {
    MergedFunc->setSubprogram(F1->getSubprogram());
  } else {
    errs() << "ERROR: different subprograms!\n";
    //MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
  }


  if (F1->getUnnamedAddr() == F2->getUnnamedAddr()) {
    MergedFunc->setUnnamedAddr(F1->getUnnamedAddr());
  } else {
    errs() << "ERROR: different unnamed addr!\n";
    MergedFunc->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
  }


  if (F1->getVisibility() == F2->getVisibility()) {
    MergedFunc->setVisibility(F1->getVisibility());
  } else {
    errs() << "ERROR: different visibility!\n";
    // MergedFunc->setLinkage(PredFunc1.getFunction()->getLinkage());
  }

  // Exception Handling requires landing pads to have the same personality
  // function
  if (F1->hasPersonalityFn() && F2->hasPersonalityFn()) {
    Constant *PersonalityFn1 = F1->getPersonalityFn();
    Constant *PersonalityFn2 = F2->getPersonalityFn();
    if (PersonalityFn1 == PersonalityFn2) {
      MergedFunc->setPersonalityFn(PersonalityFn1);
    } else {
#ifdef ENABLE_DEBUG_CODE
      PersonalityFn1->dump();
      PersonalityFn2->dump();
#endif
      errs() << "ERROR: different personality function!\n";
    }
  } else if (F1->hasPersonalityFn()) {
    errs() << "Only F1 has PersonalityFn\n";
    // TODO: check if this is valid: merge function with personality with function without it
    MergedFunc->setPersonalityFn(F1->getPersonalityFn());
  } else if (F2->hasPersonalityFn()) {
    errs() << "Only F2 has PersonalityFn\n";
    // TODO: check if this is valid: merge function with personality with function without it
    MergedFunc->setPersonalityFn(F2->getPersonalityFn());
  }

  if (F1->hasComdat() && F2->hasComdat()) {
    auto *Comdat1 = F1->getComdat();
    auto *Comdat2 = F2->getComdat();
    if (Comdat1 == Comdat2) {
      MergedFunc->setComdat(Comdat1);
    } else {
      errs() << "ERROR: different comdats!\n";
    }
  } else if (F1->hasComdat()) {
    errs() << "Only F1 has Comdat\n";
    MergedFunc->setComdat(F1->getComdat()); // TODO: check if this is valid:
                                            // merge function with comdat with
                                            // function without it
  } else if (F2->hasComdat()) {
    errs() << "Only F2 has Comdat\n";
    MergedFunc->setComdat(F2->getComdat()); // TODO: check if this is valid:
                                            // merge function with comdat with
                                            // function without it
  }

  if (F1->hasSection())
    MergedFunc->setSection(F1->getSection());

}

Function *RemoveFuncIdArg(Function *F, std::vector<Argument *> &ArgsList) {

   // Start by computing a new prototype for the function, which is the same as
   // the old function, but doesn't have isVarArg set.
   FunctionType *FTy = F->getFunctionType();

   std::vector<Type *> NewArgs;
   for (unsigned i = 1; i < ArgsList.size(); i++) {
     NewArgs.push_back(ArgsList[i]->getType());
   }
   ArrayRef<llvm::Type *> Params(NewArgs);

   //std::vector<Type *> Params(FTy->param_begin(), FTy->param_end());
   FunctionType *NFTy = FunctionType::get(FTy->getReturnType(), Params, false);
   //unsigned NumArgs = Params.size();
 
   // Create the new function body and insert it into the module...
   Function *NF = Function::Create(NFTy, F->getLinkage());

   NF->copyAttributesFrom(F);

   NF->setAlignment(F->getAlignment());
   NF->setCallingConv(F->getCallingConv());
   NF->setLinkage(F->getLinkage());
   NF->setDSOLocal(F->isDSOLocal());
   NF->setSubprogram(F->getSubprogram());
   NF->setUnnamedAddr(F->getUnnamedAddr());
   NF->setVisibility(F->getVisibility());
   // Exception Handling requires landing pads to have the same personality
   // function
   if (F->hasPersonalityFn())
     NF->setPersonalityFn(F->getPersonalityFn());
   if (F->hasComdat())
     NF->setComdat(F->getComdat());
   if (F->hasSection())
    NF->setSection(F->getSection());

   F->getParent()->getFunctionList().insert(F->getIterator(), NF);
   NF->takeName(F);
 
   // Since we have now created the new function, splice the body of the old
   // function right into the new function, leaving the old rotting hulk of the
   // function empty.
   NF->getBasicBlockList().splice(NF->begin(), F->getBasicBlockList());

   std::vector<Argument *> NewArgsList;
   for (Argument &arg : NF->args()) {
     NewArgsList.push_back(&arg);
   }

   // Loop over the argument list, transferring uses of the old arguments over to
   // the new arguments, also transferring over the names as well.  While we're at
   // it, remove the dead arguments from the DeadArguments list.
   /*
   for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(),
        I2 = NF->arg_begin(); I != E; ++I, ++I2) {
     // Move the name and users over to the new version.
     I->replaceAllUsesWith(&*I2);
     I2->takeName(&*I);
   }
   */

   for (unsigned i = 0; i<NewArgsList.size(); i++) {
     ArgsList[i+1]->replaceAllUsesWith(NewArgsList[i]);
     NewArgsList[i]->takeName(ArgsList[i+1]);
   }

   // Clone metadatas from the old function, including debug info descriptor.
   SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
   F->getAllMetadata(MDs);
   for (auto MD : MDs)
     NF->addMetadata(MD.first, *MD.second);
 
   // Fix up any BlockAddresses that refer to the function.
   F->replaceAllUsesWith(ConstantExpr::getBitCast(NF, F->getType()));
   // Delete the bitcast that we just created, so that NF does not
   // appear to be address-taken.
   NF->removeDeadConstantUsers();
   // Finally, nuke the old function.
   F->eraseFromParent();
   return NF;
}

static Value *createCastIfNeeded(Value *V, Type *DstType, IRBuilder<> &Builder, Type *IntPtrTy, const FunctionMergingOptions &Options = {});

/*
bool CodeGenerator(Value *IsFunc1, BasicBlock *EntryBB1, BasicBlock *EntryBB2, BasicBlock *PreBB,
                   std::list<std::pair<Value *, Value *>> &AlignedInsts,
                   ValueToValueMapTy &VMap, Function *MergedFunc,
Type *RetType1, Type *RetType2, Type *ReturnType, bool RequiresUnifiedReturn, LLVMContext &Context, Type *IntPtrTy, const FunctionMergingOptions &Options = {}) {
*/



void FunctionMerger::CodeGenerator::destroyGeneratedCode() {
  for (Instruction *I : CreatedInsts) {
    I->dropAllReferences();
  }
  for (Instruction *I : CreatedInsts) {
    I->eraseFromParent();
  }
  for (BasicBlock *BB : CreatedBBs) {
    BB->eraseFromParent();
  }
  CreatedInsts.clear();
  CreatedBBs.clear();
}

bool FunctionMerger::CodeGenerator::generate(std::list<std::pair<Value *, Value *>> &AlignedInsts,
                  ValueToValueMapTy &VMap,
                  const FunctionMergingOptions &Options) {

  LLVMContext &Context = *ContextPtr;
  bool RequiresFuncId = false;

  Value *RetUnifiedAddr = nullptr;
  Value *RetAddr1 = nullptr;
  Value *RetAddr2 = nullptr;
  
  BasicBlock *MergedBB = nullptr;
  BasicBlock *MergedBB1 = nullptr;
  BasicBlock *MergedBB2 = nullptr;

  std::map<BasicBlock *, BasicBlock *> TailBBs;

  std::map<SelectCacheEntry, Value *> SelectCache;
  std::map<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *> CacheBBSelect;

  
  std::set<BasicBlock*> OriginalBlocks;
  std::set<PHINode*> PHINodes;
  for (auto Pair : AlignedInsts) {
    if (Pair.first != nullptr && Pair.second != nullptr) {

      if (isa<BasicBlock>(Pair.first)) {
        BasicBlock *NewBB = BasicBlock::Create(Context, "", MergedFunc);
        CreatedBBs.insert(NewBB);

        BasicBlock *BB1 = dyn_cast<BasicBlock>(Pair.first);
        BasicBlock *BB2 = dyn_cast<BasicBlock>(Pair.second);
        VMap[BB1] = NewBB;
        VMap[BB2] = NewBB;
        OriginalBlocks.insert(NewBB);

        /*
        errs() << "Here 1!\n";
        errs() << "BB1: " << GetValueName(BB1) << "\n";
        errs() << "BB2: " << GetValueName(BB2) << "\n";
        errs() << "NewBB: " << GetValueName(NewBB) << "\n";
        */
        IRBuilder<> Builder(NewBB);
        //handling PHI nodes
        /*
        if (HandlePHINodes) {
          for (Instruction &I : *BB1) {
            if (isa<PHINode>(&I)) {
              PHINode *NewPHI = Builder.CreatePHI(I.getType(), 0);
              CreatedInsts.insert(NewPHI);
              VMap[&I] = NewPHI;
              PHINodes.insert( dyn_cast<PHINode>(&I));
            } else break;
          }
          for (Instruction &I : *BB2) {
            if (isa<PHINode>(&I)) {
              PHINode *NewPHI = Builder.CreatePHI(I.getType(), 0);
              CreatedInsts.insert(NewPHI);
              VMap[&I] = NewPHI;
              PHINodes.insert( dyn_cast<PHINode>(&I));
            } else break;
          }
        }
        */
        if (BB1->isLandingPad() || BB2->isLandingPad()) {
          LandingPadInst *LP1 = BB1->getLandingPadInst();
          LandingPadInst *LP2 = BB2->getLandingPadInst();
          assert((LP1 != nullptr && LP2 != nullptr) &&
                 "Should be both as per the BasicBlock match!");
          Instruction *NewLP = LP1->clone();
          CreatedInsts.insert(NewLP);
          VMap[LP1] = NewLP;
          VMap[LP2] = NewLP;

          Builder.Insert(NewLP);
        }
      }
    } else {
      Value *V = nullptr;
      if (Pair.first) {
        V = Pair.first;
      } else {
        V = Pair.second;
      }

      if (isa<BasicBlock>(V)) {
        BasicBlock *BB = dyn_cast<BasicBlock>(V);

        BasicBlock *NewBB = BasicBlock::Create(Context, "", MergedFunc);
        CreatedBBs.insert(NewBB);
        OriginalBlocks.insert(NewBB);

        VMap[BB] = NewBB;
        TailBBs[dyn_cast<BasicBlock>(V)] = NewBB;

        //errs() << "Here 2!\n";
        IRBuilder<> Builder(NewBB);

        /*
        if (HandlePHINodes) {
          for (Instruction &I : *BB) {
            if (isa<PHINode>(&I)) {
              PHINode *NewPHI = Builder.CreatePHI(I.getType(), 0);
              CreatedInsts.insert(NewPHI);
              VMap[&I] = NewPHI;
              PHINodes.insert( dyn_cast<PHINode>(&I));
            } else break;
          }
        }
        */
        if (BB->isLandingPad()) {
          LandingPadInst *LP = BB->getLandingPadInst();
          Instruction *NewLP = LP->clone();
          CreatedInsts.insert(NewLP);
          VMap[LP] = NewLP;

          Builder.Insert(NewLP);
        }
      }
    }
  }

  if (RequiresUnifiedReturn) {
    IRBuilder<> Builder(PreBB);
    RetUnifiedAddr = Builder.CreateAlloca(ReturnType);
    CreatedInsts.insert(dyn_cast<Instruction>(RetUnifiedAddr));

    RetAddr1 = Builder.CreateAlloca(RetType1);
    RetAddr2 = Builder.CreateAlloca(RetType2);
    CreatedInsts.insert(dyn_cast<Instruction>(RetAddr1));
    CreatedInsts.insert(dyn_cast<Instruction>(RetAddr2));
  }

  if (VMap[EntryBB1]!=VMap[EntryBB2]) {
    IRBuilder<> Builder(PreBB);
    Instruction *Br = Builder.CreateCondBr(IsFunc1, dyn_cast<BasicBlock>(VMap[EntryBB1]),
                         dyn_cast<BasicBlock>(VMap[EntryBB2]));
    CreatedInsts.insert(Br);
  } else {
    BasicBlock *NewEntryBB = dyn_cast<BasicBlock>(VMap[EntryBB1]);
    if (NewEntryBB->size()==0) {
      VMap[EntryBB1] = PreBB;
      VMap[EntryBB2] = PreBB;
      CreatedBBs.erase(NewEntryBB);
      NewEntryBB->eraseFromParent();
    } else {
      IRBuilder<> Builder(PreBB);
      Instruction *Br = Builder.CreateBr(NewEntryBB);
      CreatedInsts.insert(Br);
    }
  }


  std::set< std::pair<Value*,Value*> > CreatedTerms;
 
  for (auto Pair : AlignedInsts) {
    // mergeable instructions
    if (Pair.first != nullptr && Pair.second != nullptr) {

      if (isa<BasicBlock>(Pair.first)) {
        BasicBlock *NewBB =
            dyn_cast<BasicBlock>(VMap[dyn_cast<BasicBlock>(Pair.first)]);

        MergedBB = NewBB;
        MergedBB1 = dyn_cast<BasicBlock>(Pair.first);
        MergedBB2 = dyn_cast<BasicBlock>(Pair.second);

      } else {
        assert(isa<Instruction>(Pair.first) && "Instruction expected!");
        Instruction *I1 = dyn_cast<Instruction>(Pair.first);
        Instruction *I2 = dyn_cast<Instruction>(Pair.second);


        if (MergedBB == nullptr) {
          MergedBB = BasicBlock::Create(Context, "", MergedFunc);
          CreatedBBs.insert(MergedBB);
          {
            IRBuilder<> Builder(TailBBs[ dyn_cast<BasicBlock>(I1->getParent()) ]);
            Instruction *Br = Builder.CreateBr(MergedBB);
            CreatedInsts.insert(Br);
          }
          {
            IRBuilder<> Builder(TailBBs[ dyn_cast<BasicBlock>(I2->getParent()) ]);
            Instruction *Br = Builder.CreateBr(MergedBB);
            CreatedInsts.insert(Br);
          }
        }
        MergedBB1 = dyn_cast<BasicBlock>(I1->getParent());
        MergedBB2 = dyn_cast<BasicBlock>(I2->getParent());

        Instruction *NewI = nullptr;


        Instruction *I = I1;

        IRBuilder<> Builder(MergedBB);
        if (I1->getOpcode() == Instruction::Ret) {
          if (RequiresUnifiedReturn) {
            NewI = Builder.CreateRet(UndefValue::get(ReturnType));
          } else {
            if (I1->getNumOperands() >= I2->getNumOperands())
              I = I1;
            else
              I = I2;
            NewI = I->clone();
            Builder.Insert(NewI);
          }
        } else {
          assert(I1->getNumOperands() == I2->getNumOperands() &&
                 "Num of Operands SHOULD be EQUAL\n");
          NewI = I->clone();
          Builder.Insert(NewI);
        }

        CreatedInsts.insert(NewI);

        VMap[I1] = NewI;
        VMap[I2] = NewI;

        // TODO: temporary removal of metadata
        
        SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
        NewI->getAllMetadata(MDs);
        for (std::pair<unsigned, MDNode *> MDPair : MDs) {
          NewI->setMetadata(MDPair.first, nullptr);
        }

        if (isa<TerminatorInst>(NewI)) {
          MergedBB = nullptr;
          MergedBB1 = nullptr;
          MergedBB2 = nullptr;
        }
      }
    } else {
      RequiresFuncId = true;

      if (MergedBB != nullptr) {

        BasicBlock *NewBB1 = BasicBlock::Create(Context, "", MergedFunc);
        BasicBlock *NewBB2 = BasicBlock::Create(Context, "", MergedFunc);
        CreatedBBs.insert(NewBB1);
        CreatedBBs.insert(NewBB2);

        TailBBs[MergedBB1] = NewBB1;
        TailBBs[MergedBB2] = NewBB2;

        IRBuilder<> Builder(MergedBB);
        Instruction *Br = Builder.CreateCondBr(IsFunc1, NewBB1, NewBB2);
        CreatedInsts.insert(Br);

        MergedBB = nullptr;
      }

      Value *V = nullptr;
      if (Pair.first) {
        V = Pair.first;
      } else {
        V = Pair.second;
      }

      if (isa<BasicBlock>(V)) {
        BasicBlock *NewBB = dyn_cast<BasicBlock>(VMap[dyn_cast<BasicBlock>(V)]);
        TailBBs[dyn_cast<BasicBlock>(V)] = NewBB;
      } else {
        assert(isa<Instruction>(V) && "Instruction expected!");
        Instruction *I = dyn_cast<Instruction>(V);

        Instruction *NewI = nullptr;
        if (I->getOpcode() == Instruction::Ret && !ReturnType->isVoidTy() &&
            I->getNumOperands() == 0) {
          NewI = ReturnInst::Create(Context, UndefValue::get(ReturnType));
        } else
          NewI = I->clone();

        CreatedInsts.insert(NewI);
        VMap[I] = NewI;

        BasicBlock *BBPoint = TailBBs[dyn_cast<BasicBlock>(I->getParent())];
        if (BBPoint == nullptr) {
          BBPoint = TailBBs[dyn_cast<BasicBlock>(I->getParent())] =
              dyn_cast<BasicBlock>(VMap[dyn_cast<BasicBlock>(I->getParent())]);
        }


        IRBuilder<> Builder(BBPoint);
        Builder.Insert(NewI);

        // TODO: temporarily removing metadata
        
        SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
        NewI->getAllMetadata(MDs);
        for (std::pair<unsigned, MDNode *> MDPair : MDs) {
          NewI->setMetadata(MDPair.first, nullptr);
        }
      }

    }
  }

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen1.stopTimer();
#endif

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen2.startTimer();
#endif

  for (auto Pair : AlignedInsts) {
    // mergable instructions
    if (Pair.first != nullptr && Pair.second != nullptr) {

      if (isa<Instruction>(Pair.first)) {
        Instruction *I1 = dyn_cast<Instruction>(Pair.first);
        Instruction *I2 = dyn_cast<Instruction>(Pair.second);

        Instruction *I = I1;
        if (I1->getOpcode() == Instruction::Ret) {
          if (I1->getNumOperands() >= I2->getNumOperands())
            I = I1;
          else
            I = I2;
        } else {
          assert(I1->getNumOperands() == I2->getNumOperands() &&
                 "Num of Operands SHOULD be EQUAL\n");
        }

        Instruction *NewI = dyn_cast<Instruction>(VMap[I]);

        IRBuilder<> Builder(NewI);

        if (isa<BinaryOperator>(NewI) && I->isCommutative()) {
          //CountBinOps++;

          BinaryOperator *BO1 = dyn_cast<BinaryOperator>(I1);
          BinaryOperator *BO2 = dyn_cast<BinaryOperator>(I2);
          Value *VL1 = MapValue(BO1->getOperand(0), VMap);
          Value *VL2 = MapValue(BO2->getOperand(0), VMap);
          Value *VR1 = MapValue(BO1->getOperand(1), VMap);
          Value *VR2 = MapValue(BO2->getOperand(1), VMap);
          if (VL1 == VR2 && VL2 != VR2) {
            Value *TmpV = VR2;
            VR2 = VL2;
            VL2 = TmpV;
            //CountOpReorder++;
          }

          std::vector<std::pair<Value *, Value *>> Vs;
          Vs.push_back(std::pair<Value *, Value *>(VL1, VL2));
          Vs.push_back(std::pair<Value *, Value *>(VR1, VR2));

          for (unsigned i = 0; i < Vs.size(); i++) {
            Value *V1 = Vs[i].first;
            Value *V2 = Vs[i].second;

            Value *V = V1; // first assume that V1==V2
            if (V1 != V2) {
              RequiresFuncId = true;
              // create predicated select instruction
              if (V1 == ConstantInt::getTrue(Context) &&
                  V2 == ConstantInt::getFalse(Context)) {
                V = IsFunc1;
              } else if (V1 == ConstantInt::getFalse(Context) &&
                         V2 == ConstantInt::getTrue(Context)) {
                V = Builder.CreateNot(IsFunc1);
                CreatedInsts.insert(dyn_cast<Instruction>(V));
              } else {
                Value *SelectI = nullptr;

                SelectCacheEntry SCE(IsFunc1, V1, V2, NewI->getParent());
                if (SelectCache.find(SCE) != SelectCache.end()) {
                  SelectI = SelectCache[SCE];
                } else {
                  Value *CastedV2 =
                      createCastIfNeeded(V2, V1->getType(), Builder, IntPtrTy, Options);
                  SelectI = Builder.CreateSelect(IsFunc1, V1, CastedV2);
                  CreatedInsts.insert(dyn_cast<Instruction>(SelectI));

                  ListSelects.push_back(dyn_cast<Instruction>(SelectI));

                  SelectCache[SCE] = SelectI;
                }

                V = SelectI;
              }
            }

            //TODO: cache the created instructions
            Value *CastedV = createCastIfNeeded(
                V, NewI->getOperand(i)->getType(), Builder, IntPtrTy, Options);
            NewI->setOperand(i, CastedV);
          }
        } else if ( I->getOpcode() == Instruction::Ret && RequiresUnifiedReturn ) {
          Value *V1 = MapValue(I1->getOperand(0), VMap);
          Value *V2 = MapValue(I2->getOperand(0), VMap);

          if (V1->getType()!=ReturnType) {
            Instruction *SI = Builder.CreateStore(V1, RetAddr1);
            CreatedInsts.insert(SI);
            Value *CastedAddr = Builder.CreatePointerCast(RetAddr1, RetUnifiedAddr->getType());
            CreatedInsts.insert(dyn_cast<Instruction>(CastedAddr));
            Instruction *LI = Builder.CreateLoad(ReturnType, CastedAddr);
            CreatedInsts.insert(LI);
            V1 = LI;
          }
          if (V2->getType()!=ReturnType) {
            Instruction *SI = Builder.CreateStore(V2, RetAddr2);
            CreatedInsts.insert(SI);
            Value *CastedAddr = Builder.CreatePointerCast(RetAddr2, RetUnifiedAddr->getType());
            CreatedInsts.insert(dyn_cast<Instruction>(CastedAddr));
            Instruction *LI = Builder.CreateLoad(ReturnType, CastedAddr);
            CreatedInsts.insert(LI);
            V2 = LI;
          }
          
          Value *SelV = Builder.CreateSelect(IsFunc1, V1, V2);
          if (isa<Instruction>(SelV)) {
            CreatedInsts.insert(dyn_cast<Instruction>(SelV));
            ListSelects.push_back(dyn_cast<Instruction>(SelV));
          }

          NewI->setOperand(0,SelV);
        } else {
          for (unsigned i = 0; i < I->getNumOperands(); i++) {
            Value *F1V = nullptr;
            Value *V1 = nullptr;
            if (i < I1->getNumOperands()) {
              F1V = I1->getOperand(i);
              V1 = MapValue(I1->getOperand(i), VMap);
              assert(V1!=nullptr && "Mapped value should NOT be NULL!");
              /*
              if (V1 == nullptr) {
                errs() << "ERROR: Null value mapped: V1 = "
                          "MapValue(I1->getOperand(i), "
                          "VMap);\n";
                MergedFunc->eraseFromParent();
                return ErrorResponse;
              }
              */
            } else
              V1 = UndefValue::get(I2->getOperand(i)->getType());

            Value *F2V = nullptr;
            Value *V2 = nullptr;
            if (i < I2->getNumOperands()) {
              F2V = I2->getOperand(i);
              V2 = MapValue(I2->getOperand(i), VMap);
              assert(V2!=nullptr && "Mapped value should NOT be NULL!");
              /*
              if (V2 == nullptr) {
                errs() << "ERROR: Null value mapped: V2 = "
                          "MapValue(I2->getOperand(i), "
                          "VMap);\n";
                MergedFunc->eraseFromParent();
                return ErrorResponse;
              }
              */
            } else
              V2 = UndefValue::get(I1->getOperand(i)->getType());

            assert(V1 != nullptr && "Value should NOT be null!");
            assert(V2 != nullptr && "Value should NOT be null!");

            Value *V = V1; // first assume that V1==V2

            if (V1 != V2) {
              RequiresFuncId = true;

              //TODO: Create BasicBlock Select function
              if (isa<BasicBlock>(V1) && isa<BasicBlock>(V2)) {
                auto CacheKey = std::pair<BasicBlock *, BasicBlock *>(
                    dyn_cast<BasicBlock>(V1), dyn_cast<BasicBlock>(V2));
                BasicBlock *SelectBB = nullptr;
                if (CacheBBSelect.find(CacheKey) != CacheBBSelect.end()) {
                  SelectBB = CacheBBSelect[CacheKey];
                } else {
                  SelectBB = BasicBlock::Create(Context, "", MergedFunc);
                  IRBuilder<> BuilderBB(SelectBB);

                  CreatedBBs.insert(SelectBB);

                  BasicBlock *BB1 = dyn_cast<BasicBlock>(V1);
                  BasicBlock *BB2 = dyn_cast<BasicBlock>(V2);

                  if (BB1->isLandingPad() || BB2->isLandingPad()) {
                    LandingPadInst *LP1 = BB1->getLandingPadInst();
                    LandingPadInst *LP2 = BB2->getLandingPadInst();
                    assert ( (LP1!=nullptr && LP2!=nullptr) && "Should be both as per the BasicBlock match!");

                    Instruction *NewLP = LP1->clone();
                    BuilderBB.Insert(NewLP);

                    CreatedInsts.insert(NewLP);
                    
                    BasicBlock *F1BB = dyn_cast<BasicBlock>(F1V);
                    BasicBlock *F2BB = dyn_cast<BasicBlock>(F2V);

                    VMap[F1BB] = SelectBB;
                    VMap[F2BB] = SelectBB;
                    if (TailBBs[F1BB]==nullptr) TailBBs[F1BB]=BB1;
                    if (TailBBs[F2BB]==nullptr) TailBBs[F2BB]=BB2;
                    VMap[F1BB->getLandingPadInst()] = NewLP;
                    VMap[F2BB->getLandingPadInst()] = NewLP;
                    
                    BB1->replaceAllUsesWith(SelectBB);
                    BB2->replaceAllUsesWith(SelectBB);

                    //remove landingpad instructions from 
                    LP1->replaceAllUsesWith(NewLP);
                    CreatedInsts.erase(LP1);
                    LP1->eraseFromParent();
                    LP2->replaceAllUsesWith(NewLP);
                    CreatedInsts.erase(LP2);
                    LP2->eraseFromParent();
                  }

                  Instruction *Br = BuilderBB.CreateCondBr(IsFunc1, BB1, BB2);
                  CreatedInsts.insert(Br);
                  CacheBBSelect[CacheKey] = SelectBB;
                }
                V = SelectBB;
              } else {

                //TODO: Create Value Select function
                // create predicated select instruction
                if (V1 == ConstantInt::getTrue(Context) &&
                    V2 == ConstantInt::getFalse(Context)) {
                  V = IsFunc1;
                } else if (V1 == ConstantInt::getFalse(Context) &&
                           V2 == ConstantInt::getTrue(Context)) {
                  V = Builder.CreateNot(IsFunc1);
                  CreatedInsts.insert(dyn_cast<Instruction>(V));
                } else {
                  Value *SelectI = nullptr;

                  SelectCacheEntry SCE(IsFunc1, V1, V2, NewI->getParent());
                  if (SelectCache.find(SCE) != SelectCache.end()) {
                    SelectI = SelectCache[SCE];
                  } else {
                    //TODO: cache created instructions
                    Value *CastedV2 = createCastIfNeeded(V2, V1->getType(),
                                                         Builder, IntPtrTy, Options);
                    SelectI = Builder.CreateSelect(IsFunc1, V1, CastedV2);
                    CreatedInsts.insert(dyn_cast<Instruction>(SelectI));

                    ListSelects.push_back(dyn_cast<Instruction>(SelectI));

                    SelectCache[SCE] = SelectI;
                  }

                  V = SelectI;
                }
              }
            }

            Value *CastedV = V;
            if (!isa<BasicBlock>(V)) {
              //TODO: cache the created instructions
              CastedV = createCastIfNeeded(V, NewI->getOperand(i)->getType(),
                                           Builder, IntPtrTy, Options);
            }
            NewI->setOperand(i, CastedV);
          }
        } // end of commutative if-else


      }

    } else {
      RequiresFuncId = true;

      bool isFuncId1 = true;
      Value *V = nullptr;
      if (Pair.first) {
        isFuncId1 = true;
        V = Pair.first;
      } else {
        isFuncId1 = false;
        V = Pair.second;
      }

      if (isa<Instruction>(V)) {
        Instruction *I = dyn_cast<Instruction>(V);

        Instruction *NewI = dyn_cast<Instruction>(VMap[I]);

        IRBuilder<> Builder(NewI);

        if ( I->getOpcode() == Instruction::Ret && RequiresUnifiedReturn ) {
          Value *V = MapValue(I->getOperand(0), VMap);
          
          if (V->getType()!=ReturnType) {
            Value *Addr = (isFuncId1?RetAddr1:RetAddr2);
            Instruction *SI = Builder.CreateStore(V, Addr);

            CreatedInsts.insert(SI);

            Value *CastedAddr = Builder.CreatePointerCast( Addr, RetUnifiedAddr->getType() );
            CreatedInsts.insert(dyn_cast<Instruction>(CastedAddr));


            Instruction *LI = Builder.CreateLoad(ReturnType, CastedAddr);
            CreatedInsts.insert(LI);

            V = LI;
          }

          NewI->setOperand(0,V);
        } else {
          for (unsigned i = 0; i < I->getNumOperands(); i++) {
            Value *V = MapValue(I->getOperand(i), VMap);
            assert( V!=nullptr && "Mapped value should NOT be NULL!");
            /*
            if (V == nullptr) {
              errs() << "ERROR: Null value mapped: V = "
                        "MapValue(I->getOperand(i), VMap);\n";
              MergedFunc->eraseFromParent();
              return ErrorResponse;
            }
            */
            Value *CastedV = V;
            if (!isa<BasicBlock>(V))
              CastedV = createCastIfNeeded(V, NewI->getOperand(i)->getType(),
                                           Builder, IntPtrTy, Options);
            NewI->setOperand(i, CastedV);
          }
        }
      }
    }
  }

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen2.stopTimer();
#endif

  /*
  if (HandlePHINodes) {
    //DominatorTree DT(*MergedFunc);
  
    for (PHINode *PHI : PHINodes) {
      PHINode *NewPHI = dyn_cast<PHINode>(VMap[PHI]);
      BasicBlock *TargetBB = dyn_cast<BasicBlock>(NewPHI->getParent());
      unsigned NumPreds = 0;
      for (auto It = pred_begin(TargetBB), E=pred_end(TargetBB); It!=E; It++) {
        NewPHI->addIncoming(UndefValue::get(NewPHI->getType()), *It);
        NumPreds++;
      }
  
      for (unsigned i = 0; i<PHI->getNumIncomingValues(); i++) {
        Instruction *NewBI = dyn_cast<Instruction>(MapValue(PHI->getIncomingBlock(i)->getTerminator(), VMap));
        BasicBlock *MappedBB = NewBI->getParent();
        Value *MappedV = MapValue(PHI->getIncomingValue(i), VMap);
        for (unsigned j = 0; j<NewPHI->getNumIncomingValues(); j++) {
          BasicBlock *InBB = NewPHI->getIncomingBlock(j);
          if (InBB==MappedBB) {
            NewPHI->setIncomingValue(j,MappedV);
          }
        }
      }


    }
  }
  */

  //TODO: if RequiresFuncId==false, remove the first argument of the merged function
/*
#ifdef TIME_STEPS_DEBUG
  TimeCodeGenFix.startTimer();
#endif
  {
    DominatorTree DT(*MergedFunc);
    removeRedundantInstructions(DT, ListSelects);

    if (!fixNotDominatedUses(MergedFunc, DT)) {
      MergedFunc->eraseFromParent();
      MergedFunc = nullptr;
      CreatedInsts.insert(SI);
    }
  }
#ifdef TIME_STEPS_DEBUG
  TimeCodeGenFix.stopTimer();
#endif
*/
  return RequiresFuncId;
}

static void simplifySelects(Function *F) {
  std::set<SelectInst*> AllSelects;

  for (Instruction &I : instructions(F)) {
    if (isa<SelectInst>(&I)) {
      AllSelects.insert(dyn_cast<SelectInst>(&I));
    }
  }

  for (SelectInst *SI : AllSelects) {
    if (isa<AllocaInst>(SI->getTrueValue()) && isa<AllocaInst>(SI->getFalseValue())) {
      Instruction *TrueI = dyn_cast<Instruction>(SI->getTrueValue());
      Instruction *FalseI = dyn_cast<Instruction>(SI->getFalseValue());
      if (TrueI->getNumUses()==1 && FalseI->getNumUses()==1) {
        SI->replaceAllUsesWith(TrueI);
        SI->dropAllReferences();
        SI->eraseFromParent();
        FalseI->eraseFromParent();
      }
    }
  }
}

bool FunctionMerger::CodeGenerator::commit() {
#ifdef TIME_STEPS_DEBUG
  TimeCodeGenFix.startTimer();
#endif
  {
    DominatorTree DT(*MergedFunc);
    removeRedundantInstructions(DT, ListSelects);

    if (!fixNotDominatedUses(MergedFunc, PreBB, DT)) {
      return false;
    }
    simplifySelects(MergedFunc);

    bool FoundFixedPoint = false;
    while (!FoundFixedPoint) {
      FoundFixedPoint = true;
      for (BasicBlock &BB: *MergedFunc) {
        BranchInst *BI = dyn_cast<BranchInst>(BB.getTerminator());
        if (BB.size()==1 && BI!=nullptr && (!BI->isConditional()) && BB.getSinglePredecessor()!=nullptr) {
          BasicBlock *PredBB = BB.getSinglePredecessor();
          BranchInst *PredBI = dyn_cast<BranchInst>(PredBB->getTerminator());
          if (PredBI) {
            for (unsigned i = 0; i<PredBI->getNumSuccessors(); i++) {
              if (PredBI->getSuccessor(i)==(&BB)) {
                PredBI->setSuccessor(i,BI->getSuccessor(0));
              }
            }
            BI->eraseFromParent();
            BB.eraseFromParent();
            FoundFixedPoint = false;
            break;
          }
        }
        /*
        if (TryToSimplifyUncondBranchFromEmptyBlock(&BB)) {
          FoundFixedPoint = false;
          break;
        }
        */
      }
    }
  }
#ifdef TIME_STEPS_DEBUG
  TimeCodeGenFix.stopTimer();
#endif
  return true;
}

FunctionMergeResult FunctionMerger::merge(Function *F1, Function *F2, const FunctionMergingOptions &Options) {
  LLVMContext &Context = *ContextPtr;
  FunctionMergeResult ErrorResponse(F1, F2, nullptr);

  if (!validMergePair(F1,F2))
    return ErrorResponse;

  SmallVector<Value*,8> F1Vec;
  SmallVector<Value*,8> F2Vec;
  linearize(F1, F1Vec);
  linearize(F2, F2Vec);

#ifdef TIME_STEPS_DEBUG
  TimeAlign.startTimer();
#endif

  ScoringSystem Scoring;
  Scoring.setMatchProfit(1)
         .setAllowMismatch(false)
         .setGapStartPenalty(-3)
         .setGapExtendPenalty(0)
         .setPenalizeStartingGap(true)
         .setPenalizeEndingGap(false);

  SequenceAligner<Value*> SA(F1Vec,F2Vec,FunctionMerger::match,(Value*)nullptr,Scoring);
  std::list<std::pair<Value *, Value *>> &AlignedInsts = SA.Result.Data;

#ifdef TIME_STEPS_DEBUG
  TimeAlign.stopTimer();
#endif

#ifdef ENABLE_DEBUG_CODE
  if (Verbose) {
    for (auto Pair : AlignedInsts) {
      if (Pair.first != nullptr && Pair.second != nullptr) {
        errs() << "1: ";
        if (isa<BasicBlock>(Pair.first))
          errs() << "BB " << GetValueName(Pair.first) << "\n";
        else
          Pair.first->dump();
        errs() << "2: ";
        if (isa<BasicBlock>(Pair.second))
          errs() << "BB " << GetValueName(Pair.second) << "\n";
        else
          Pair.second->dump();
        errs() << "----\n";
      } else {
        if (Pair.first) {
          errs() << "1: ";
          if (isa<BasicBlock>(Pair.first))
            errs() << "BB " << GetValueName(Pair.first) << "\n";
          else
            Pair.first->dump();
          errs() << "2: -\n";
        } else if (Pair.second) {
          errs() << "1: -\n";
          errs() << "2: ";
          if (isa<BasicBlock>(Pair.second))
            errs() << "BB " << GetValueName(Pair.second) << "\n";
          else
            Pair.second->dump();
        }
        errs() << "----\n";
      }
    }
  }
#endif

#ifdef TIME_STEPS_DEBUG
  TimeParam.startTimer();
#endif

  errs() << "Creating function type\n";

  // Merging parameters
  std::map<unsigned, unsigned> ParamMap1;
  std::map<unsigned, unsigned> ParamMap2;
  std::vector<Type *> Args;


  errs() << "Merging arguments\n";
  MergeArguments(Context, F1, F2, AlignedInsts, ParamMap1,ParamMap2,Args,Options);

  Type *RetType1 = F1->getReturnType();
  Type *RetType2 = F2->getReturnType();
  Type *ReturnType = nullptr;

  bool RequiresUnifiedReturn = false;

  //Value *RetUnifiedAddr = nullptr;
  //Value *RetAddr1 = nullptr;
  //Value *RetAddr2 = nullptr;
  
  if (validMergeTypes(F1, F2, Options)) {
    errs() << "Simple return types\n";
    ReturnType = RetType1;
    if (ReturnType->isVoidTy()) {
      ReturnType = RetType2;
    }
  } else if (Options.EnableUnifiedReturnType) {
    errs() << "Unifying return types\n";
    RequiresUnifiedReturn = true;

    auto SizeOfTy1 = DL->getTypeStoreSize(RetType1);
    auto SizeOfTy2 = DL->getTypeStoreSize(RetType2);
    if (SizeOfTy1 >= SizeOfTy2) {
      ReturnType = RetType1;
    } else {
      ReturnType = RetType2;
    }
  } else return ErrorResponse;

  FunctionType *FTy = FunctionType::get(ReturnType, ArrayRef<Type*>(Args), false);
  Function *MergedFunc =
      Function::Create(FTy, GlobalValue::LinkageTypes::InternalLinkage,
                       Twine("m"), M);


  errs() << "Initializing VMap\n";
  ValueToValueMapTy VMap;

  std::vector<Argument *> ArgsList;
  for (Argument &arg : MergedFunc->args()) {
    ArgsList.push_back(&arg);
  }
  Value *FuncId = ArgsList[0];

  int ArgId = 0;
  for (auto I = F1->arg_begin(), E = F1->arg_end(); I != E; I++) {
    VMap[&(*I)] = ArgsList[ParamMap1[ArgId]];
    ArgId++;
  }
  ArgId = 0;
  for (auto I = F2->arg_begin(), E = F2->arg_end(); I != E; I++) {
    VMap[&(*I)] = ArgsList[ParamMap2[ArgId]];
    ArgId++;
  }

#ifdef TIME_STEPS_DEBUG
  TimeParam.stopTimer();
#endif

  errs() << "Setting attributes\n";
  SetFunctionAttributes(F1,F2,MergedFunc);

  Value *IsFunc1 = FuncId;

#ifdef TIME_STEPS_DEBUG
  TimeCodeGen1.startTimer();
#endif

  errs() << "Running code generator\n";


  CodeGenerator CG(ContextPtr, IntPtrTy);
  CG.setFunctionIdentifier(IsFunc1)
    .setEntryPoints(&F1->getEntryBlock(), &F2->getEntryBlock())
    .setReturnTypes(RetType1,RetType2)
    .setMergedFunction(MergedFunc)
    .setMergedEntryPoint(BasicBlock::Create(Context, "", MergedFunc))
    .setMergedReturnType(ReturnType, RequiresUnifiedReturn);
  bool RequiresFuncId = CG.generate(AlignedInsts, VMap, Options);
  CG.commit();

  if (!RequiresFuncId) {
    errs() << "Removing FuncId\n";
    /*
    MergedFunc = RemoveFuncIdArg(MergedFunc, ArgsList);

    for (auto &kv : ParamMap1) {
      ParamMap1[kv.first] = kv.second - 1;
    }
    for (auto &kv : ParamMap2) {
      ParamMap2[kv.first] = kv.second - 1;
    }
    FuncId = nullptr;
    */
  }


  FunctionMergeResult Result(F1, F2, MergedFunc, RequiresUnifiedReturn);
  Result.setArgumentMapping(F1, ParamMap1);
  Result.setArgumentMapping(F2, ParamMap2);
  Result.setFunctionIdArgument(FuncId != nullptr);
  return Result;
}

static bool canReplaceAllCalls(Function *F) {
  for (User *U : F->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledFunction() != F)
        return false;
    } else
      return false;
  }
  return true;
}

void FunctionMerger::replaceByCall(Function *F, FunctionMergeResult &MFR, const FunctionMergingOptions &Options) {
  LLVMContext &Context = M->getContext();

  Value *FuncId = MFR.getFunctionIdValue(F);
  Function *MergedF = MFR.getMergedFunction();

  F->deleteBody();
  BasicBlock *NewBB = BasicBlock::Create(Context, "", F);
  IRBuilder<> Builder(NewBB);

  std::vector<Value *> args;
  for (unsigned i = 0; i < MergedF->getFunctionType()->getNumParams(); i++) {
    args.push_back(nullptr);
  }

  if (MFR.hasFunctionIdArgument()) {
    args[0] = FuncId;
  }

  std::vector<Argument *> ArgsList;
  for (Argument &arg : F->args()) {
    ArgsList.push_back(&arg);
  }

  for (auto Pair : MFR.getArgumentMapping(F)) {
    args[Pair.second] = ArgsList[Pair.first];
  }

  for (unsigned i = 0; i < args.size(); i++) {
    if (args[i] == nullptr) {
      args[i] = UndefValue::get(MergedF->getFunctionType()->getParamType(i));
    }
  }

  CallInst *CI =
      (CallInst *)Builder.CreateCall(MergedF, ArrayRef<Value *>(args));
  CI->setTailCall();
  CI->setCallingConv(MergedF->getCallingConv());
  CI->setAttributes(MergedF->getAttributes());
  CI->setIsNoInline();

  if (F->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Value *CastedV = CI;
    if (MFR.needUnifiedReturn()) {
      Value *AddrCI = Builder.CreateAlloca(CI->getType());
      Builder.CreateStore(CI,AddrCI);
      Value *CastedAddr = Builder.CreatePointerCast(AddrCI, PointerType::get(F->getReturnType(), DL->getAllocaAddrSpace()));
      CastedV = Builder.CreateLoad(CastedAddr);
    } else {
      CastedV = createCastIfNeeded(CI, F->getReturnType(), Builder, IntPtrTy, Options);
    }
    Builder.CreateRet(CastedV);
  }
}

bool FunctionMerger::replaceCallsWith(Function *F, FunctionMergeResult &MFR, const FunctionMergingOptions &Options) {

  Value *FuncId = MFR.getFunctionIdValue(F);
  Function *MergedF = MFR.getMergedFunction();

  std::vector<CallInst *> Calls;
  for (User *U : F->users()) {
    if (CallInst *CI = dyn_cast<CallInst>(U)) {
      if (CI->getCalledFunction() == F) {
        CallInst *CI = dyn_cast<CallInst>(U); // CS.getInstruction());
        Calls.push_back(CI);
      } else
        return false;
    } else
      return false;
  }

  for (CallInst *CI : Calls) {
    IRBuilder<> Builder(CI);

    std::vector<Value *> args;
    for (unsigned i = 0; i < MergedF->getFunctionType()->getNumParams(); i++) {
      args.push_back(nullptr);
    }

    if (MFR.hasFunctionIdArgument()) {
      args[0] = FuncId;
    }

    for (auto Pair : MFR.getArgumentMapping(F)) {
      args[Pair.second] = CI->getArgOperand(Pair.first);
    }

    for (unsigned i = 0; i < args.size(); i++) {
      if (args[i] == nullptr) {
        args[i] = UndefValue::get(MergedF->getFunctionType()->getParamType(i));
      }
    }

    CallInst *NewCI = (CallInst *)Builder.CreateCall(MergedF->getFunctionType(),
                                                     MergedF, args);
    NewCI->setCallingConv(MergedF->getCallingConv());
    NewCI->setAttributes(MergedF->getAttributes());
    NewCI->setIsNoInline();

    Value *CastedV = NewCI;
    if (!F->getReturnType()->isVoidTy()) {
      if (MFR.needUnifiedReturn()) {
        Value *AddrCI = Builder.CreateAlloca(NewCI->getType());
        Builder.CreateStore(NewCI,AddrCI);
        Value *CastedAddr = Builder.CreatePointerCast(AddrCI, PointerType::get(F->getReturnType(), DL->getAllocaAddrSpace()));
        CastedV = Builder.CreateLoad(CastedAddr);
      } else {
        CastedV = createCastIfNeeded(NewCI, F->getReturnType(), Builder, IntPtrTy, Options);
      }
    }

    // if (F->getReturnType()==MergedF->getReturnType())
    if (CI->getNumUses() > 0) {
      CI->replaceAllUsesWith(CastedV);
    }

    if (CI->getNumUses() == 0) {
      CI->eraseFromParent();
    } else {
      if (CI->getNumUses() > 0) {

        if (Verbose) {
          errs() << "ERROR: Function Call has uses\n";
#ifdef ENABLE_DEBUG_CODE
          CI->dump();
          errs() << "Called type\n";
          F->getReturnType()->dump();
          errs() << "Merged type\n";
          MergedF->getReturnType()->dump();
#endif
        }
      }
    }
  }

  return true;
}

static bool ShouldPreserveGV(const GlobalValue *GV) {
  // Function must be defined here
  if (GV->isDeclaration())
    return true;

  // Available externally is really just a "declaration with a body".
  //if (GV->hasAvailableExternallyLinkage())
  //  return true;

  // Assume that dllexported symbols are referenced elsewhere
  if (GV->hasDLLExportStorageClass())
    return true;

  // Already local, has nothing to do.
  if (GV->hasLocalLinkage())
    return false;

  return false;
}

void FunctionMerger::updateCallGraph(FunctionMergeResult &MFR, StringSet<> &AlwaysPreserved, const FunctionMergingOptions &Options) {
  auto FPair = MFR.getFunctions();
  Function *F1 = FPair.first;
  Function *F2 = FPair.second;

  replaceByCall(F1, MFR, Options);
  replaceByCall(F2, MFR, Options);

  bool CanEraseF1 = replaceCallsWith(F1, MFR, Options);
  bool CanEraseF2 = replaceCallsWith(F2, MFR, Options);

  if (CanEraseF1 && (F1->getNumUses() == 0)
     && (HasWholeProgram?true:ShouldPreserveGV(F1))
     && (AlwaysPreserved.find(F1->getName()) == AlwaysPreserved.end())) {
    F1->eraseFromParent();
  }

  if (CanEraseF2 && (F2->getNumUses() == 0)
     && (HasWholeProgram?true:ShouldPreserveGV(F2))
     && (AlwaysPreserved.find(F2->getName()) == AlwaysPreserved.end())) {
    F2->eraseFromParent();
  }
}

int requiresOriginalInterfaces(FunctionMergeResult &MFR) {
  auto FPair = MFR.getFunctions();
  return (canReplaceAllCalls(FPair.first) ? 0 : 1) +
         (canReplaceAllCalls(FPair.second) ? 0 : 1);
}


static bool compareFunctionScores(const std::pair<Function *, unsigned> &F1,
                                  const std::pair<Function *, unsigned> &F2) {
  return F1.second > F2.second;
}

//#define FMSA_USE_JACCARD

class Fingerprint {
public:
  static const size_t MaxOpcode = 65;
  int OpcodeFreq[MaxOpcode];
  // std::map<unsigned, int> OpcodeFreq;
  // size_t NumOfInstructions;
  // size_t NumOfBlocks;

  #ifdef FMSA_USE_JACCARD
  std::set<Type *> Types;
  #else
  std::map<Type*, int> TypeFreq;
  #endif

  Function *F;

  Fingerprint(Function *F) {
    this->F = F;

    memset(OpcodeFreq, 0, sizeof(int) * MaxOpcode);
    // for (int i = 0; i<MaxOpcode; i++) OpcodeFreq[i] = 0;

    // NumOfInstructions = 0;
    for (Instruction &I : instructions(F)) {
      OpcodeFreq[I.getOpcode()]++;
      /*
            if (OpcodeFreq.find(I.getOpcode()) != OpcodeFreq.end())
              OpcodeFreq[I.getOpcode()]++;
            else
              OpcodeFreq[I.getOpcode()] = 1;
      */
      // NumOfInstructions++;

      
      #ifdef FMSA_USE_JACCARD
      Types.insert(I.getType());
      #else
      TypeFreq[I.getType()]++;
      #endif
    }
    // NumOfBlocks = F->size();
  }
};

class FingerprintSimilarity {
public:
  Function *F1;
  Function *F2;
  int Similarity;
  int LeftOver;
  int TypesDiff;
  int TypesSim;
  float Score;

  FingerprintSimilarity() : F1(nullptr), F2(nullptr), Score(0.0f) {}

  FingerprintSimilarity(Fingerprint *FP1, Fingerprint *FP2) {
    F1 = FP1->F;
    F2 = FP2->F;

    Similarity = 0;
    LeftOver = 0;
    TypesDiff = 0;
    TypesSim = 0;

    for (unsigned i = 0; i < Fingerprint::MaxOpcode; i++) {
      int Freq1 = FP1->OpcodeFreq[i];
      int Freq2 = FP2->OpcodeFreq[i];
      int MinFreq = std::min(Freq1, Freq2);
      Similarity += MinFreq;
      LeftOver += std::max(Freq1, Freq2) - MinFreq;
    }
    
    #ifdef FMSA_USE_JACCARD
    for (auto Ty1 : FP1->Types) {
      if (FP2->Types.find(Ty1) == FP2->Types.end())
        TypesDiff++;
      else
        TypesSim++;
    }
    for (auto Ty2 : FP2->Types) {
      if (FP1->Types.find(Ty2) == FP1->Types.end())
        TypesDiff++;
    }

    float TypeScore = ((float)TypesSim) / ((float)TypesSim + TypesDiff);
    #else
    for (auto Pair : FP1->TypeFreq) {
      if (FP2->TypeFreq.find(Pair.first) == FP2->TypeFreq.end()) {
        TypesDiff += Pair.second;
      } else {
        int MinFreq = std::min(Pair.second, FP2->TypeFreq[Pair.first]);
        TypesSim += MinFreq;
        TypesDiff +=
            std::max(Pair.second, FP2->TypeFreq[Pair.first]) - MinFreq;
      }
    }
    for (auto Pair : FP2->TypeFreq) {
      if (FP1->TypeFreq.find(Pair.first) == FP1->TypeFreq.end()) {
        TypesDiff += Pair.second;
      }
    }
    float TypeScore =
        ((float)TypesSim) / ((float)(TypesSim * 2.0f + TypesDiff));
    #endif
    float UpperBound =
        ((float)Similarity) / ((float)(Similarity * 2.0f + LeftOver));

    #ifdef FMSA_USE_JACCARD
    Score = UpperBound * TypeScore;
    #else
    Score = std::min(UpperBound,TypeScore);
    #endif
  }

  bool operator<(const FingerprintSimilarity &FS) const {
    return Score < FS.Score;
  }

  bool operator>(const FingerprintSimilarity &FS) const {
    return Score > FS.Score;
  }

  bool operator<=(const FingerprintSimilarity &FS) const {
    return Score <= FS.Score;
  }

  bool operator>=(const FingerprintSimilarity &FS) const {
    return Score >= FS.Score;
  }

  bool operator==(const FingerprintSimilarity &FS) const {
    return Score == FS.Score;
  }
};

bool SimilarityHeuristicFilter(const FingerprintSimilarity &Item) {
  if (!ApplySimilarityHeuristic)
    return true;

  if (Item.Similarity < Item.LeftOver)
    return false;

  float TypesDiffRatio = (((float)Item.TypesDiff) / ((float)Item.TypesSim));
  if (TypesDiffRatio > 1.5f)
    return false;

  return true;
}

int EstimateFunctionSize(Function *F, TargetTransformInfo *TTI) {
  int size = 0;
  for (Instruction &I : instructions(F)) {
    if (isa<AllocaInst>(&I)) continue;
    size += TTI->getInstructionCost(
        &I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
  }
  return size;
}

#ifdef TIME_STEPS_DEBUG
Timer TimePreProcess("Merge::Preprocess", "Merge::Preprocess");
Timer TimeLin("Merge::Lin", "Merge::Lin");
Timer TimeRank("Merge::Rank", "Merge::Rank");
Timer TimeUpdate("Merge::Update", "Merge::Update");
#endif

bool FunctionMerging::runOnModule(Module &M) {
  StringSet<> AlwaysPreserved;
  AlwaysPreserved.insert("main");

  srand(time(NULL));


  FunctionMergingOptions Options = FunctionMergingOptions()
                                    .maximizeParameterScore(MaxParamScore)
                                    .matchOnlyIdenticalTypes(IdenticalType)
                                    .enableUnifiedReturnTypes(EnableUnifiedReturnType);

  auto *PSI = this->getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI(); //TODO getPSI returns PSI *
  auto LookupBFI = [this](Function &F) {
    return &this->getAnalysis<BlockFrequencyInfoWrapperPass>(F).getBFI();
  };

  //TODO: We could use a TTI ModulePass instead but current TTI analysis pass is
  //a FunctionPass.
  TargetTransformInfo TTI(M.getDataLayout());

  std::vector<std::pair<Function *, unsigned>> FunctionsToProcess;

  unsigned TotalOpReorder = 0;
  unsigned TotalBinOps = 0;

  FunctionMerger FM(&M,PSI,LookupBFI);

  std::map<Function *, Fingerprint *> CachedFingerprints;
  std::map<Function *, unsigned> FuncSizes;

#ifdef TIME_STEPS_DEBUG
  TimePreProcess.startTimer();
#endif

  for (auto &F : M) {
    if (F.isDeclaration() || F.isVarArg())
      continue;

    FuncSizes[&F] = EstimateFunctionSize(&F, &TTI);

    
    if (!HandlePHINodes) {
      demoteRegToMem(F);
    }
    
    FunctionsToProcess.push_back(
      std::pair<Function *, unsigned>(&F, FuncSizes[&F]) );

    CachedFingerprints[&F] = new Fingerprint(&F);
  }

  std::sort(FunctionsToProcess.begin(), FunctionsToProcess.end(),
            compareFunctionScores);

#ifdef TIME_STEPS_DEBUG
  TimePreProcess.stopTimer();
#endif

  std::list<Function *> WorkList;

  std::set<Function *> AvailableCandidates;
  for (std::pair<Function *, unsigned> FuncAndSize1 : FunctionsToProcess) {
    Function *F1 = FuncAndSize1.first;
    WorkList.push_back(F1);
    AvailableCandidates.insert(F1);
  }

  std::vector<FingerprintSimilarity> Rank;
  if (ExplorationThreshold > 1)
    Rank.reserve(FunctionsToProcess.size());

  FunctionsToProcess.clear();

  while (!WorkList.empty()) {
    Function *F1 = WorkList.front();
    WorkList.pop_front();

    AvailableCandidates.erase(F1);

    Rank.clear();

#ifdef TIME_STEPS_DEBUG
    TimeRank.startTimer();
#endif

    Fingerprint *FP1 = CachedFingerprints[F1];

    if (ExplorationThreshold > 1) {
      for (Function *F2 : AvailableCandidates) {
        if ((!FM.validMergeTypes(F1, F2, Options) && !Options.EnableUnifiedReturnType) || !validMergePair(F1, F2))
          continue;

        Fingerprint *FP2 = CachedFingerprints[F2];

        FingerprintSimilarity PairSim(FP1, FP2);
        if (SimilarityHeuristicFilter(PairSim))
          Rank.push_back(PairSim);
      }
      std::make_heap(Rank.begin(), Rank.end());
    } else {

      bool FoundCandidate = false;
      FingerprintSimilarity BestPair;

      for (Function *F2 : AvailableCandidates) {
        if ((!FM.validMergeTypes(F1, F2, Options) && !Options.EnableUnifiedReturnType) || !validMergePair(F1, F2))
          continue;

        Fingerprint *FP2 = CachedFingerprints[F2];

        FingerprintSimilarity PairSim(FP1, FP2);
        if (PairSim > BestPair && SimilarityHeuristicFilter(PairSim)) {
          BestPair = PairSim;
          FoundCandidate = true;
        }
      }
      if (FoundCandidate)
        Rank.push_back(BestPair);
    }

    unsigned MergingTrialsCount = 0;

    while (!Rank.empty()) {
      auto RankEntry = Rank.front();
      Function *F2 = RankEntry.F2;
      std::pop_heap(Rank.begin(), Rank.end());
      Rank.pop_back();

      //CountBinOps = 0;
      //CountOpReorder = 0;

      MergingTrialsCount++;

      if (Debug || Verbose) {
        errs() << "Attempting: " << GetValueName(F1) << ", " << GetValueName(F2)
               << "\n";
      }

      FunctionMergeResult Result = FM.merge(F1,F2,Options);

      bool validFunction = true;

      if (Result.getMergedFunction() != nullptr && verifyFunction(*Result.getMergedFunction())) {
        if (Debug || Verbose) {
          errs() << "Invalid Function: " << GetValueName(F1) << ", "
                 << GetValueName(F2) << "\n";
        }
#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          if (Result.getMergedFunction() != nullptr) {
            Result.getMergedFunction()->dump();
          }
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
        }
#endif
        Result.getMergedFunction()->eraseFromParent();
        validFunction = false;
      }

      if (Result.getMergedFunction() && validFunction) {
        DominatorTree MergedDT(*Result.getMergedFunction());
        promoteMemoryToRegister(*Result.getMergedFunction(), MergedDT);
        //demoteRegToMem(*Result.getMergedFunction());

        unsigned SizeF1 = FuncSizes[F1];
        unsigned SizeF2 = FuncSizes[F2];

        unsigned SizeF12 = requiresOriginalInterfaces(Result) * 3 +
                           EstimateFunctionSize(Result.getMergedFunction(), &TTI);

#ifdef ENABLE_DEBUG_CODE
        if (Verbose) {
          errs() << "F1:\n";
          F1->dump();
          errs() << "F2:\n";
          F2->dump();
          errs() << "F1-F2:\n";
          Result.getMergedFunction()->dump();
        }
#endif

        if (Debug || Verbose) {
          errs() << "Sizes: " << SizeF1 << " + " << SizeF2 << " <= " << SizeF12 << "?\n";
        }

        if (Debug || Verbose) {
          errs() << "Estimated reduction: "
                 << (int)((1 - ((double)SizeF12) / (SizeF1 + SizeF2)) * 100)
                 << "% ("
                 << (SizeF12 < (SizeF1 + SizeF2) *
                                   ((100.0 + MergingOverheadThreshold) / 100.0))
                 << ") " << MergingTrialsCount << " : " << GetValueName(F1)
                 << "; " << GetValueName(F2) << " | Score " << RankEntry.Score
                 << "\n";
        }

        if (SizeF12 <
            (SizeF1 + SizeF2) * ((100.0 + MergingOverheadThreshold) / 100.0)) {

          //MergingDistance.push_back(MergingTrialsCount);

          //TotalOpReorder += CountOpReorder;
          //TotalBinOps += CountBinOps;

          if (Debug || Verbose) {
            errs() << "Merged: " << GetValueName(F1) << ", " << GetValueName(F2)
                   << " = " << GetValueName(Result.getMergedFunction()) << "\n";
          }

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.startTimer();
#endif

          AvailableCandidates.erase(F2);
          WorkList.remove(F2);

          FM.updateCallGraph(Result, AlwaysPreserved, Options);

          // feed new function back into the working lists
          WorkList.push_front(Result.getMergedFunction());
          AvailableCandidates.insert(Result.getMergedFunction());

          FuncSizes[Result.getMergedFunction()] =
              EstimateFunctionSize(Result.getMergedFunction(), &TTI);

          //TODO: demote phi instructions
          if (!HandlePHINodes) {
            demoteRegToMem(*Result.getMergedFunction());
          }
          CachedFingerprints[Result.getMergedFunction()] =
              new Fingerprint(Result.getMergedFunction());

#ifdef TIME_STEPS_DEBUG
          TimeUpdate.stopTimer();
#endif

          break; // end exploration

        } else {
          if (Result.getMergedFunction() != nullptr)
            Result.getMergedFunction()->eraseFromParent();
        }
      }

      if (MergingTrialsCount >= ExplorationThreshold) {
        break;
      }
    }
  }

  WorkList.clear();

  for (auto kv : CachedFingerprints) {
    delete kv.second;
  }
  CachedFingerprints.clear();

  double MergingAverageDistance = 0;
  unsigned MergingMaxDistance = 0;

  if (Debug || Verbose) {
    errs() << "Total operand reordering: " << TotalOpReorder << "/"
           << TotalBinOps << " ("
           << 100.0 * (((double)TotalOpReorder) / ((double)TotalBinOps))
           << " %)\n";

//    errs() << "Total parameter score: " << TotalParamScore << "\n";

//    errs() << "Total number of merges: " << MergingDistance.size() << "\n";
    errs() << "Average number of trials before merging: "
           << MergingAverageDistance << "\n";
    errs() << "Maximum number of trials before merging: " << MergingMaxDistance
           << "\n";
  }

#ifdef TIME_STEPS_DEBUG
  errs() << "Timer:Align: " << TimeAlign.getTotalTime().getWallTime() << "\n";
  TimeAlign.clear();

  errs() << "Timer:Param: " << TimeParam.getTotalTime().getWallTime() << "\n";
  TimeParam.clear();

  errs() << "Timer:CodeGen1: " << TimeCodeGen1.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGen1.clear();

  errs() << "Timer:CodeGen2: " << TimeCodeGen2.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGen2.clear();

  errs() << "Timer:CodeGenFix: " << TimeCodeGenFix.getTotalTime().getWallTime()
         << "\n";
  TimeCodeGenFix.clear();

  errs() << "Timer:PreProcess: " << TimePreProcess.getTotalTime().getWallTime()
         << "\n";
  TimePreProcess.clear();

  errs() << "Timer:Lin: " << TimeLin.getTotalTime().getWallTime() << "\n";
  TimeLin.clear();

  errs() << "Timer:Rank: " << TimeRank.getTotalTime().getWallTime() << "\n";
  TimeRank.clear();

  errs() << "Timer:Update: " << TimeUpdate.getTotalTime().getWallTime() << "\n";
  TimeUpdate.clear();
#endif

  return true;
}

void FunctionMerging::getAnalysisUsage(AnalysisUsage &AU) const {
  ModulePass::getAnalysisUsage(AU);
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
}

char FunctionMerging::ID = 0;
INITIALIZE_PASS(FunctionMerging, "func-merging", "New Function Merging", false,
                false)

ModulePass *llvm::createFunctionMergingPass() {
  return new FunctionMerging();
}

static std::string GetValueName(const Value *V) {
  if (V) {
    std::string name;
    raw_string_ostream namestream(name);
    V->printAsOperand(namestream, false);
    return namestream.str();
  } else
    return "[null]";
}

/// Create a cast instruction if needed to cast V to type DstType. We treat
/// pointer and integer types of the same bitwidth as equivalent, so this can be
/// used to cast them to each other where needed. The function returns the Value
/// itself if no cast is needed, or a new CastInst instance inserted before
/// InsertBefore. The integer type equivalent to pointers must be passed as
/// IntPtrType (get it from DataLayout). This is guaranteed to generate no-op
/// casts, otherwise it will assert.
//Value *FunctionMerger::createCastIfNeeded(Value *V, Type *DstType, IRBuilder<> &Builder, const FunctionMergingOptions &Options) {
Value *createCastIfNeeded(Value *V, Type *DstType, IRBuilder<> &Builder, Type *IntPtrTy, const FunctionMergingOptions &Options) {

  if (V->getType() == DstType || Options.IdenticalTypesOnly)
    return V;

  Value *Result;
  Type *OrigType = V->getType();

  if (OrigType->isStructTy()) {
    assert(DstType->isStructTy());
    assert(OrigType->getStructNumElements() == DstType->getStructNumElements());

    Result = UndefValue::get(DstType);
    for (unsigned int I = 0, E = OrigType->getStructNumElements(); I < E; ++I) {
      Value *ExtractedValue =
          Builder.CreateExtractValue(V, ArrayRef<unsigned int>(I));
      Value *Element =
          createCastIfNeeded(ExtractedValue, DstType->getStructElementType(I),
                             Builder, IntPtrTy, Options);
      Result =
          Builder.CreateInsertValue(Result, Element, ArrayRef<unsigned int>(I));
    }
    return Result;
  }
  assert(!DstType->isStructTy());

  if (OrigType->isPointerTy() &&
      (DstType->isIntegerTy() || DstType->isPointerTy())) {
    Result = Builder.CreatePointerCast(V, DstType, "merge_cast");
  } else if (OrigType->isIntegerTy() && DstType->isPointerTy() &&
             OrigType == IntPtrTy) {
    // Int -> Ptr
    Result = Builder.CreateCast(CastInst::IntToPtr, V, DstType, "merge_cast");
  } else {
    llvm_unreachable("Can only cast int -> ptr or ptr -> (ptr or int)");
  }

  // assert(cast<CastInst>(Result)->isNoopCast(InsertAtEnd->getParent()->getParent()->getDataLayout())
  // &&
  //    "Cast is not a no-op cast. Potential loss of precision");

  return Result;
}

static bool valueEscapes(const Instruction *Inst) {
  const BasicBlock *BB = Inst->getParent();
  for (const User *U : Inst->users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (UI->getParent() != BB || isa<PHINode>(UI))
      return true;
  }
  return false;
}


//TODO: use the function implemented by the reg2mem pass directly
//-reg2mem
static void demoteRegToMem(Function &F) {
  if (F.isDeclaration())
    return;

  // Insert all new allocas into entry block.
  BasicBlock *BBEntry = &F.getEntryBlock();

  assert(pred_empty(BBEntry) &&
         "Entry block to function must not have predecessors!");

  // Find first non-alloca instruction and create insertion point. This is
  // safe if block is well-formed: it always have terminator, otherwise
  // we'll get and assertion.
  BasicBlock::iterator I = BBEntry->begin();
  while (isa<AllocaInst>(I))
    ++I;

  CastInst *AllocaInsertionPoint = new BitCastInst(
      Constant::getNullValue(Type::getInt32Ty(F.getContext())),
      Type::getInt32Ty(F.getContext()), "reg2mem alloca point", &*I);

  // Find the escaped instructions. But don't create stack slots for
  // allocas in entry block.
  std::list<Instruction *> WorkList;
  for (BasicBlock &ibb : F)
    for (BasicBlock::iterator iib = ibb.begin(), iie = ibb.end(); iib != iie;
         ++iib) {
      if (!(isa<AllocaInst>(iib) && iib->getParent() == BBEntry) &&
          valueEscapes(&*iib)) {
        WorkList.push_front(&*iib);
      }
    }

  // Demote escaped instructions
  // NumRegsDemoted += WorkList.size();
  for (Instruction *ilb : WorkList)
    DemoteRegToStack(*ilb, false, AllocaInsertionPoint);

  WorkList.clear();

  // Find all phi's
  for (BasicBlock &ibb : F)
    for (BasicBlock::iterator iib = ibb.begin(), iie = ibb.end(); iib != iie;
         ++iib)
      if (isa<PHINode>(iib))
        WorkList.push_front(&*iib);

  // Demote phi nodes
  // NumPhisDemoted += WorkList.size();
  for (Instruction *ilb : WorkList)
    DemotePHIToStack(cast<PHINode>(ilb), AllocaInsertionPoint);
}

//TODO: use the function implemented by the mem2reg pass directly
static bool
promoteMemoryToRegister(Function &F,
                        DominatorTree &DT) { //, AssumptionCache &AC) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    // PromoteMemToReg(Allocas, DT, &AC);
    PromoteMemToReg(Allocas, DT, nullptr);
    // NumPromoted += Allocas.size();
    Changed = true;
  }
  return Changed;
}


static bool fixNotDominatedUses(Function *F, BasicBlock *Entry, DominatorTree &DT) {

  std::list<Instruction *> WorkList;
  std::map<Instruction *, Value *> StoredAddress;

  std::map< Instruction *, std::map< Instruction *, std::list<unsigned> > >
      UpdateList;

  bool HasPHINodes = false;
  for (Instruction &I : instructions(*F)) {
    for (auto *U : I.users()) {
      Instruction *UI = dyn_cast<Instruction>(U);
      if (UI && !DT.dominates(&I, UI)) {
        auto &ListOperands = UpdateList[&I][UI];
        for (unsigned i = 0; i < UI->getNumOperands(); i++) {
          if (UI->getOperand(i) == (Value *)(&I)) {
            ListOperands.push_back(i);
          }
        }
      }
    }
    if (UpdateList[&I].size() > 0) {
      IRBuilder<> Builder(&*Entry->getFirstInsertionPt());
      StoredAddress[&I] = Builder.CreateAlloca(I.getType());
      //Builder.CreateStore(GetAnyValue(I.getType()), StoredAddress[&I]);
      Value *V = &I;
      if (I.getParent()->getTerminator()) {
        InvokeInst *II = dyn_cast<InvokeInst>(I.getParent()->getTerminator());
        if ((&I)==I.getParent()->getTerminator() && II!=nullptr) {
          BasicBlock *SrcBB = I.getParent();
          BasicBlock *DestBB = II->getNormalDest();
          Builder.SetInsertPoint(DestBB->getFirstNonPHI());
          //create PHI
          if (DestBB->getSinglePredecessor()==nullptr) {
            PHINode *PHI = Builder.CreatePHI( I.getType(), 0 );
            HasPHINodes = true;
            for (auto it = pred_begin(DestBB), et = pred_end(DestBB); it != et; ++it) {
              BasicBlock *BB = *it;
              if (BB==SrcBB) {
                PHI->addIncoming(&I,BB);
              } else {
                PHI->addIncoming( UndefValue::get(I.getType()) ,BB);
              }
            }
            V = PHI;
          }
        } else {
          Builder.SetInsertPoint(I.getParent()->getTerminator());
        }
      } else {
        Builder.SetInsertPoint(I.getParent());
      }
      Builder.CreateStore(V, StoredAddress[&I]);
    }
  }


  for (auto &kv1 : UpdateList) {
    Instruction *I = kv1.first;
    if (kv1.second.size()>0) {
      auto End = kv1.second.end();
      auto It = kv1.second.begin();

      std::vector<Instruction*> Users;

      BasicBlock *CommonBB = (*It).first->getParent();

      Users.push_back((*It).first);

      It++;

      while (It!=End) {
        BasicBlock *BB = DT.findNearestCommonDominator(CommonBB, (*It).first->getParent());

        if (BB!=nullptr) {
          Users.push_back((*It).first);
          CommonBB = BB;
        } else {
          IRBuilder<> Builder(&*CommonBB->getFirstInsertionPt());
          Value *V = Builder.CreateLoad(StoredAddress[I]);
          for (Instruction *UI : Users) {
            for (unsigned i : kv1.second[UI]) {
              UI->setOperand(i, V);
            }
          }
          Users.clear();
          CommonBB = (*It).first->getParent();
          Users.push_back((*It).first);
        }
          
        It++;
      }

    }

  }

  if (HasPHINodes) demoteRegToMem(*F);

  return true;
}


void FunctionMerger::CodeGenerator::removeRedundantInstructions(DominatorTree &DT,
                                   std::vector<Instruction *> &ListInsts) {
  std::set<Instruction *> SkipList;

  std::map<Instruction *, std::list<Instruction *>> UpdateList;

  for (Instruction *I1 : ListInsts) {
    if (SkipList.find(I1) != SkipList.end())
      continue;
    for (Instruction *I2 : ListInsts) {
      if (I1 == I2)
        continue;
      if (SkipList.find(I2) != SkipList.end())
        continue;
      assert(I1->getNumOperands() == I2->getNumOperands() &&
             "Should have the same num of operands!");
      bool AllEqual = true;
      for (unsigned i = 0; i < I1->getNumOperands(); ++i) {
        AllEqual = AllEqual && (I1->getOperand(i) == I2->getOperand(i));
      }

      if (AllEqual && DT.dominates(I1, I2)) {
        UpdateList[I1].push_back(I2);
        SkipList.insert(I2);
        SkipList.insert(I1);
      }
    }
  }

  int count = 0;
  for (auto &kv : UpdateList) {
    for (auto *I : kv.second) {
      count++;
      CreatedInsts.erase(I);
      I->replaceAllUsesWith(kv.first);
      I->eraseFromParent();
    }
  }
  errs() << "RedundantsErased: "<< count << "\n";
}
