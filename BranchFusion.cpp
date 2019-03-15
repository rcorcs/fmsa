//===- BranchFusion.cpp - A branch fusion pass ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/BranchFusion.h"

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

#define DEBUG_TYPE "BranchFusion"
//#define ENABLE_DEBUG_CODE

//#define FMSA_USE_JACCARD

//#define TIME_STEPS_DEBUG

using namespace llvm;
static std::string GetValueName(const Value *V) {
  if (V) {
    std::string name;
    raw_string_ostream namestream(name);
    V->printAsOperand(namestream, false);
    return namestream.str();
  } else
    return "[null]";
}

void BranchFusionLegacyPass::getAnalysisUsage(AnalysisUsage &AU) const {
  //AU.addRequired<DominatorTreeWrapperPass>();
  //AU.addRequired<PostDominatorTreeWrapperPass>();
  //AU.addRequired<TargetTransformInfoWrapperPass>();
}

void linearizeDominatedBlocks(BasicBlock *BB, BasicBlock *DomBB, SmallVectorImpl<Value*> &CFG, DominatorTree &DT, std::set<Value*> &Visited) {
  if (DT.dominates(DomBB,BB)) {
    if (Visited.find(BB)!=Visited.end()) return;
    Visited.insert(BB);
    
    CFG.push_back(BB);
    for (Instruction &I : *BB) {
      CFG.push_back(&I);
    }

    for (auto BBIt = succ_begin(BB), EndIt = succ_end(BB); BBIt!=EndIt; BBIt++) {
      linearizeDominatedBlocks(*BBIt,DomBB,CFG,DT,Visited);
    }
  }
}

void linearizeDominatedBlocks(BasicBlock *DomBB, SmallVectorImpl<Value*> &CFG, DominatorTree &DT) {
  std::set<Value*> Visited;
  linearizeDominatedBlocks(DomBB,DomBB,CFG,DT,Visited);
}

bool merge(Function &F, BranchInst *BI, DominatorTree &DT, TargetTransformInfo &TTI, std::list<BranchInst*> &ListBIs) {
  bool HasMerged = false;

  BasicBlock *BBT = BI->getSuccessor(0);
  BasicBlock *BBF = BI->getSuccessor(1);
  Value *BrCond = BI->getCondition();

  SmallVector<Value*, 8> CFGLeft;
  SmallVector<Value*, 8> CFGRight;

  linearizeDominatedBlocks(BBT,CFGLeft,DT);
  linearizeDominatedBlocks(BBF,CFGRight,DT);

  ScoringSystem Scoring;
  Scoring.setMatchProfit(2)
         .setAllowMismatch(false)
         .setGapStartPenalty(-3)
         .setGapExtendPenalty(-1)
         .setPenalizeStartingGap(true)
         .setPenalizeEndingGap(false);

  SequenceAligner<Value*> SA(CFGLeft,CFGRight,FunctionMerger::match,nullptr,Scoring);

  std::list<std::pair<Value *, Value *>> AlignedInsts = SA.Result.Data;

  int CountMatchUsefullInsts = 0;
  for (auto Pair : AlignedInsts) {
    if (Pair.first != nullptr && Pair.second != nullptr) {
      if (isa<BinaryOperator>(Pair.first))
        CountMatchUsefullInsts++;
      if (isa<CallInst>(Pair.first))
        CountMatchUsefullInsts++;
      if (isa<InvokeInst>(Pair.first))
        CountMatchUsefullInsts++;
      if (isa<CmpInst>(Pair.first))
        CountMatchUsefullInsts++;
      if (isa<CastInst>(Pair.first))
        CountMatchUsefullInsts++;
      //if (isa<StoreInst>(Pair.first))
      //  CountMatchUsefullInsts++;
    }
  }


  LLVMContext &Context = F.getContext();
  const DataLayout *DL = &F.getParent()->getDataLayout();
  Type *IntPtrTy = DL->getIntPtrType(Context);

  int SizeLeft = 0;
  int SizeRight = 0;

  ValueToValueMapTy VMap;
  //initialize VMap
  for (Argument &Arg : F.args()) {
    VMap[&Arg] = &Arg;
  }
  std::set<BasicBlock*> KnownBBs;
  for (Value *V : CFGLeft) {
    if (isa<BasicBlock>(V))
      KnownBBs.insert(dyn_cast<BasicBlock>(V));
    else if (Instruction *I = dyn_cast<Instruction>(V)) {
      SizeLeft += TTI.getInstructionCost(
        I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
    }
  }
  for (Value *V : CFGRight) {
    if (isa<BasicBlock>(V))
      KnownBBs.insert(dyn_cast<BasicBlock>(V));
    else if (Instruction *I = dyn_cast<Instruction>(V)) {
      SizeRight += TTI.getInstructionCost(
        I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
    }
  }
  for (BasicBlock &BB : F) {
    if (KnownBBs.count(&BB)) continue;
    VMap[&BB] = &BB;
    for (Instruction &I : BB) {
      VMap[&I] = &I;
    }
  }

  FunctionMergingOptions Options = FunctionMergingOptions()
                                    .enableUnifiedReturnTypes(false)
                                    .matchOnlyIdenticalTypes(true);

  BasicBlock *EntryBB = BasicBlock::Create(Context, "", &F);

  FunctionMerger::CodeGenerator CG(&Context, IntPtrTy);
  CG.setFunctionIdentifier(BrCond)
    .setEntryPoints(BBT, BBF)
    .setReturnTypes(F.getReturnType(),F.getReturnType())
    .setMergedFunction(&F)
    .setMergedEntryPoint(EntryBB)
    .setMergedReturnType(F.getReturnType(), false);

  CG.generate(AlignedInsts, VMap, Options);
//  CG.commit();

  //F.dump();

  int MergedSize = 0;
  errs() << "Computing size...\n";
  for (Instruction *I: CG) {
    MergedSize += TTI.getInstructionCost(
        I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
  }
  
  errs() << "SizeLeft: " << SizeLeft << "\n";
  errs() << "SizeRight: " << SizeRight << "\n";
  errs() << "Original Size: " << (SizeLeft + SizeRight) << "\n";
  errs() << "New Size: " << MergedSize << "\n";

  errs() << "SizeDiff: " << (SizeLeft + SizeRight) << " X " << MergedSize << "\n";


  if (MergedSize > SizeLeft + SizeRight) {
    errs() << "Destroying generated code\n";

    CG.destroyGeneratedCode();
    EntryBB->eraseFromParent();

  } else {
    float Profit = ((float)(SizeLeft + SizeRight)-MergedSize)/((float)SizeLeft + SizeRight);
    errs() << "Destroying original code: " << (SizeLeft + SizeRight) << " X " << MergedSize << ": " << ((int)(Profit*100.0)) << "% Reduction [" << CountMatchUsefullInsts << "] : " << GetValueName(&F) << "\n";

    IRBuilder<> Builder(BI);
    Instruction *NewBI = Builder.CreateBr(EntryBB);
    BI->eraseFromParent();


      std::vector<Instruction *> DeadInsts;
      for (BasicBlock *BB : KnownBBs) {
        for (Instruction &I : *BB) {
          I.dropAllReferences();
          DeadInsts.push_back(&I);
        }
      }   
      for (Instruction *I : DeadInsts) {
        if (BranchInst *BI = dyn_cast<BranchInst>(I)) {
          ListBIs.remove(BI);
        }
        I->eraseFromParent();
      }
      for (BasicBlock *BB : KnownBBs) {
        BB->eraseFromParent();
      }

    CG.commit();
    HasMerged = true;
  }

  return HasMerged;
}

bool BranchFusionLegacyPass::runOnFunction(Function &F) {
  if (F.isDeclaration()) return false;

  PostDominatorTree PDT(F);
  TargetTransformInfo TTI(F.getParent()->getDataLayout());

  std::vector<BranchInst *> BIs;

  errs() << "Processing: " << GetValueName(&F) << "\n";

  int SizeBefore = 0;
  for (Instruction &I : instructions(&F)) {
      SizeBefore += TTI.getInstructionCost(
        &I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
  }

  for (BasicBlock &BB : F) {
    if (BB.getTerminator()==nullptr) continue;

    BranchInst *BI = dyn_cast<BranchInst>(BB.getTerminator());

    if (BI!=nullptr && BI->isConditional()) {

      BasicBlock *BBT = BI->getSuccessor(0);
      BasicBlock *BBF = BI->getSuccessor(1);

      // check if this branch has a triangle shape
      //      bb1
      //      |  \
      //      |  bb2
      //      |  /
      //      bb3
      if (PDT.dominates(BBT,BBF) || PDT.dominates(BBF,BBT)) continue;

      // otherwise, we can collect the sub-CFGs for each branch and merge them
      //       bb1
      //      /   \
      //     bb2  bb3
      //     ...  ...
      //     
      
      //keep track of BIs
      BIs.push_back(BI);
    }
  }

  {
    DominatorTree DT(F);
    DominatorTree *DTPtr = &DT;
    auto SortRuleLambda = [DTPtr](const Instruction *I1, const Instruction *I2) -> bool {
      if (DTPtr->dominates(I1,I2)==DTPtr->dominates(I2,I1)) return (I1<I2);
      else return !(DTPtr->dominates(I1,I2));
    };
    std::sort(BIs.begin(),BIs.end(), SortRuleLambda);
  }


  std::list<BranchInst*> ListBIs;
  for (BranchInst *BI: BIs) {
    ListBIs.push_back(BI);
  }

  bool Changed = false;
  while (!ListBIs.empty()) {
    BranchInst *BI = ListBIs.front();
    ListBIs.pop_front();
    DominatorTree DT(F);
    Changed = Changed || merge(F, BI, DT, TTI, ListBIs);
  }

  if (Changed) {
    int SizeAfter = 0;
    for (Instruction &I : instructions(&F)) {
        SizeAfter += TTI.getInstructionCost(
          &I, TargetTransformInfo::TargetCostKind::TCK_CodeSize);
    }
    errs() << "FuncSize " << GetValueName(&F) << ": " << SizeBefore << " - " << SizeAfter << " = " << (SizeBefore-SizeAfter) << "\n";
  }

  return true;
}

char BranchFusionLegacyPass::ID = 0;
INITIALIZE_PASS(BranchFusionLegacyPass, "branch-fusion", "Fuse branches to reduce code size", false, false)
