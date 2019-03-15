//===-- llvm/ADT/Hashing.h - Utilities for hashing --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SEQUENCE_ALIGNMENT_H
#define LLVM_ADT_SEQUENCE_ALIGNMENT_H

#ifdef NOT_IN_LLVM
#define SmallVectorImpl std::vector
#else
#include "llvm/ADT/SmallVector.h"
#endif

#include <cassert>
#include <cmath>
#include <list>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits.h>


//#define DEBUG_SEQALIGN

#define ScoreSystemType  int
#define SCORE_MIN INT_MIN

#define MIN3(V1,V2,V3) (std::min(V1,std::min(V2,V3)))
#define MAX3(V1,V2,V3) (std::max(V1,std::max(V2,V3)))

#define MIN4(V1,V2,V3,V4) (std::min(V1,std::min(V2,std::min(V3,V4))))
#define MAX4(V1,V2,V3,V4) (std::max(V1,std::max(V2,std::max(V3,V4))))

#ifdef NOT_IN_LLVM
#else
namespace llvm{
#endif

// Store alignment result here
template<typename Ty>
class AlignedSequence {
public:
  std::list< std::pair<Ty,Ty> > Data;
  ScoreSystemType Score;
};

class ScoringSystem {
private:

  ScoreSystemType Match;

  bool AllowMismatch;
  ScoreSystemType Mismatch;

  ScoreSystemType GapStart;
  ScoreSystemType GapExtend;

  // Needleman Wunsch only
  // Turn these off to turn off penalties for gaps at the start/end of alignment
  bool PenalizeStartingGap;
  bool PenalizeEndingGap;

  //Use this restriction to avoid gaps in one of the sequences
  enum GapRestrictionKind { GRK_NONE, GRK_NO_GAP_IN_SEQ1, GRK_NO_GAP_IN_SEQ2 }; 
  GapRestrictionKind GapRestriction;

  ScoreSystemType MinPenalty, MaxPenalty; // min, max {match/mismatch,gapopen etc.}


  void updateMinMax() {
    MinPenalty = MIN4(Match, Mismatch,GapStart+GapExtend,GapExtend);
    MaxPenalty = MAX4(Match, Mismatch,GapStart+GapExtend,GapExtend);
  }

public:
  ScoringSystem() :
    Match(1), AllowMismatch(true), Mismatch(-1), GapStart(0), GapExtend(-1),
    PenalizeStartingGap(true), PenalizeEndingGap(true), GapRestriction(GRK_NONE)
  {  
    updateMinMax();
  }

  ScoringSystem &setMatchProfit(ScoreSystemType MP) {
    Match = MP;
    updateMinMax();
    return *this;
  }

  ScoringSystem &setAllowMismatch(bool AM) {
    AllowMismatch = AM;
    return *this;
  }

  ScoringSystem &setMismatchPenalty(ScoreSystemType MP) {
    Mismatch = MP;
    updateMinMax();
    return *this;
  }

  ScoringSystem &setGapStartPenalty(ScoreSystemType GS) {
    GapStart = GS;
    updateMinMax();
    return *this;
  }

  ScoringSystem &setGapExtendPenalty(ScoreSystemType GE) {
    GapExtend = GE;
    updateMinMax();
    return *this;
  }

  ScoringSystem &setPenalizeStartingGap(bool PSG) {
    PenalizeStartingGap = PSG;
    return *this;
  }

  ScoringSystem &setPenalizeEndingGap(bool PEG) {
    PenalizeEndingGap = PEG;
    return *this;
  }

  ScoringSystem &setGapRestriction(GapRestrictionKind GRK) {
    GapRestriction = GRK;
    return *this;
  }

  ScoreSystemType getMatchProfit() {
    return Match;
  }

  bool getAllowMismatch() {
    return AllowMismatch;
  }

  ScoreSystemType getMismatchPenalty() {
    return Mismatch;
  }

  ScoreSystemType getGapStartPenalty() {
    return GapStart;
  }

  ScoreSystemType getGapExtendPenalty() {
    return GapExtend;
  }

  bool getPenalizeStartingGap() {
    return PenalizeStartingGap;
  }

  bool getPenalizeEndingGap() {
    return PenalizeEndingGap;
  }

  GapRestrictionKind getGapRestriction() {
    return GapRestriction;
  }

  ScoreSystemType getMinPenalty() { return MinPenalty; }
  ScoreSystemType getMaxPenalty() { return MaxPenalty; }
};


template<typename Ty>
class SequenceAligner {
private:
  
  std::function<bool(Ty,Ty)> Match;
  Ty Blank;

  SmallVectorImpl<Ty> &Seq1;
  SmallVectorImpl<Ty> &Seq2;

  ScoringSystem Scoring;

  unsigned ScoreWidth;
  unsigned ScoreHeight;
  
  ScoreSystemType *MatchMatrix;
  ScoreSystemType *GapIn2Matrix;
  ScoreSystemType *GapIn1Matrix;


  // Matrix names
  enum MatrixKind { MK_MATCH,MK_GAP_2,MK_GAP_1 };

  void fillMatrices(bool isSmithWaterman);
  void reverseMove(MatrixKind *CurrMatrixPtr, ScoreSystemType *CurrScorePtr, int *iPtr, int *jPtr);
  void computeResult(AlignedSequence<Ty> &Result);

public:
  AlignedSequence<Ty> Result;

  SequenceAligner( SmallVectorImpl<Ty> &Seq1, SmallVectorImpl<Ty> &Seq2,
        std::function<bool(Ty,Ty)> Match, Ty Blank, ScoringSystem &Scoring)
        : Match(Match), Blank(Blank), Seq1(Seq1), Seq2(Seq2), Scoring(Scoring) {

    ScoreHeight = Seq1.size() + 1;
    ScoreWidth = Seq2.size() + 1;

    MatchMatrix = new ScoreSystemType[ScoreHeight*ScoreWidth];
    GapIn2Matrix = new ScoreSystemType[ScoreHeight*ScoreWidth];
    GapIn1Matrix = new ScoreSystemType[ScoreHeight*ScoreWidth];

    #ifdef DEBUG_SEQALIGN
    errs() << "Filling Matrices...\n";
    #endif
    fillMatrices(false);

    #ifdef DEBUG_SEQALIGN
    errs() << "Print: MatchMatrix\n";
    for (int i = 0; i<ScoreHeight; i++) {
      for (int j = 0; j<ScoreWidth; j++) {
        errs() << MatchMatrix[i*ScoreWidth + j];
        if (j==(ScoreWidth-1)) errs() << "\t";
      }
      errs() << "\n";
    }
    errs() << "\n";
    #endif


    #ifdef DEBUG_SEQALIGN
    errs() << "Print: GapIn2Matrix\n";
    for (int i = 0; i<ScoreHeight; i++) {
      for (int j = 0; j<ScoreWidth; j++) {
        errs() << GapIn2Matrix[i*ScoreWidth + j];
        if (j==(ScoreWidth-1)) errs() << "\t";
      }
      errs() << "\n";
    }
    errs() << "\n";
    #endif

    #ifdef DEBUG_SEQALIGN
    errs() << "Print: GapIn1Matrix\n";
    for (int i = 0; i<ScoreHeight; i++) {
      for (int j = 0; j<ScoreWidth; j++) {
        errs() << GapIn1Matrix[i*ScoreWidth + j];
        if (j==(ScoreWidth-1)) errs() << "\t";
      }
      errs() << "\n";
    }
    errs() << "\n";
    #endif

    #ifdef DEBUG_SEQALIGN
    errs() << "Computing Results...\n";
    #endif
    computeResult(Result);

    delete []MatchMatrix;
    delete []GapIn2Matrix;
    delete []GapIn1Matrix;

  }

};


template<typename Ty>
void SequenceAligner<Ty>::fillMatrices(bool isSmithWaterman) {
  const ScoreSystemType Min = isSmithWaterman?0:(SCORE_MIN+std::abs(Scoring.getMinPenalty()));

  MatchMatrix[0] = 0;
  GapIn2Matrix[0] = 0;
  GapIn1Matrix[0] = 0;

  if (isSmithWaterman) {
    for(size_t j = 1; j < ScoreWidth; j++)
      MatchMatrix[j] = GapIn2Matrix[j] = GapIn1Matrix[j] = 0;
    for(size_t i = 1; i < ScoreHeight; i++) {
      size_t Index = i*ScoreWidth + 0;
      MatchMatrix[Index] = GapIn2Matrix[Index] = GapIn1Matrix[Index] = Min;
    }      
  } else {
     
    // work along first row -> [i][0]
    for(size_t i = 1; i < ScoreHeight; i++) {
      size_t Index = i*ScoreWidth + 0;
      MatchMatrix[Index] = Min;
      GapIn1Matrix[Index] = Scoring.getPenalizeStartingGap()
                           ? Scoring.getGapStartPenalty() + i * Scoring.getGapExtendPenalty()
                           : 0;
      GapIn2Matrix[Index] = Min;
    }

    // work down first column -> [0][j]
    for(size_t j = 1; j < ScoreWidth; j++) {
      MatchMatrix[j] = Min;
      GapIn1Matrix[j] = Min;
      GapIn2Matrix[j] = Scoring.getPenalizeStartingGap()
                           ? Scoring.getGapStartPenalty() + j * Scoring.getGapExtendPenalty()
                           : 0;
    }

  }

  //errs() << "fill matrix...\n";
  //errs() << "Seq1 size: " << Seq1.size() << "\n";
  //errs() << "Seq2 size: " << Seq2.size() << "\n";
  // start at position [1][1]
  for(size_t i = 1; i < ScoreHeight; i++) {
    for(size_t j = 1; j < ScoreWidth; j++) {
      size_t Index = i*ScoreWidth + j;
      size_t IndexLeft = i*ScoreWidth + (j-1);
      size_t IndexUp = (i-1)*ScoreWidth + j;
      size_t IndexUpLeft = (i-1)*ScoreWidth + (j-1);
 
      bool IsMatch = Match(Seq1[i-1],Seq2[j-1]);
      if (!Scoring.getAllowMismatch() && !IsMatch) {
        MatchMatrix[Index] = Min;
      } else {

        auto Score = Scoring.getMatchProfit();
        if (!IsMatch) Score = Scoring.getMismatchPenalty();

        // substitution
        // 1) continue alignment
        // 2) close gap in seq_a
        // 3) close gap in seq_b
        MatchMatrix[Index]
          = MAX4(MatchMatrix[IndexUpLeft] + Score,
                 GapIn2Matrix[IndexUpLeft] + Score,
                 GapIn1Matrix[IndexUpLeft] + Score,
                 Min);
      }

      // Update gap_a_scores[i][j] from position [i][j-1]
      if(i==(ScoreHeight-1) && !Scoring.getPenalizeEndingGap()) {
        GapIn2Matrix[Index] = MAX3(MatchMatrix[IndexUp],
                                   GapIn2Matrix[IndexUp],
                                   GapIn1Matrix[IndexUp]);
      } else if(Scoring.getGapRestriction()!=ScoringSystem::GapRestrictionKind::GRK_NO_GAP_IN_SEQ1
                 || i==(ScoreHeight-1)) {
        GapIn2Matrix[Index]
          = MAX4(MatchMatrix[IndexUp] + Scoring.getGapStartPenalty() + Scoring.getGapExtendPenalty(),
                 GapIn2Matrix[IndexUp] + Scoring.getGapExtendPenalty(),
                 GapIn1Matrix[IndexUp] + Scoring.getGapStartPenalty() + Scoring.getGapExtendPenalty(),
                 Min);
      } else GapIn2Matrix[Index] = Min;

      // Update gap_b_scores[i][j] from position [i-1][j]
      if(j==(ScoreWidth-1) && !Scoring.getPenalizeEndingGap()) {
        GapIn1Matrix[Index] = MAX3(MatchMatrix[IndexLeft],
                                   GapIn2Matrix[IndexLeft],
                                   GapIn1Matrix[IndexLeft]);
      } else if(Scoring.getGapRestriction()!=ScoringSystem::GapRestrictionKind::GRK_NO_GAP_IN_SEQ2
                || j==(ScoreWidth-1)) {
        GapIn1Matrix[Index]
          = MAX4(MatchMatrix[IndexLeft] + Scoring.getGapStartPenalty() + Scoring.getGapExtendPenalty(),
                 GapIn2Matrix[IndexLeft] + Scoring.getGapStartPenalty() + Scoring.getGapExtendPenalty(),
                 GapIn1Matrix[IndexLeft] + Scoring.getGapExtendPenalty(),
                 Min);
      } else GapIn1Matrix[Index] = Min;

    }
  }

}

template<typename Ty>
void SequenceAligner<Ty>::computeResult(AlignedSequence<Ty> &Result) {
  // Get max score (and therefore current matrix)
  MatrixKind CurrMatrix = MK_MATCH;
  ScoreSystemType CurrScore = MatchMatrix[ScoreWidth*ScoreHeight - 1];

  int Index = (ScoreHeight-1)*ScoreWidth + ScoreWidth - 1;
  if(GapIn2Matrix[Index] > CurrScore) {
    CurrMatrix = MK_GAP_2;
    CurrScore = GapIn2Matrix[Index];
  }

  if(GapIn1Matrix[Index] > CurrScore) {
    CurrMatrix = MK_GAP_1;
    CurrScore = GapIn1Matrix[Index];
  }

  Result.Score = CurrScore;

  int i = ScoreHeight - 1;
  int j = ScoreWidth - 1;

  while(i > 0 && j > 0) {

    switch(CurrMatrix) {
      case MK_MATCH:
        Result.Data.push_front( std::pair<Ty,Ty>(Seq1[i-1],Seq2[j-1]) );
        break;
      case MK_GAP_1:
        Result.Data.push_front( std::pair<Ty,Ty>(Blank,Seq2[j-1]) );
       break;
      case MK_GAP_2:
        Result.Data.push_front( std::pair<Ty,Ty>(Seq1[i-1],Blank) );
        break;
      default:
        assert(false && "Should NOT be here!");
    }

    if(i > 0 && j > 0) {
      reverseMove(&CurrMatrix, &CurrScore, &i, &j);
    }

  }
 
  //errs() << "Processing trailing gaps\n";

  // Gap in A
  while(i > 0) {
    Result.Data.push_front( std::pair<Ty,Ty>(Seq1[i-1],Blank) );
    i--;
  }

  // Gap in B
  while(j > 0) {
    Result.Data.push_front( std::pair<Ty,Ty>(Blank,Seq2[j-1]) );
    j--;
  }
  
}

// Backtrack through scoring matrices
template<typename Ty>
void SequenceAligner<Ty>::reverseMove(MatrixKind *CurrMatrixPtr, ScoreSystemType *CurrScorePtr, int *iPtr, int *jPtr) {
  //errs() << "Reversing move...\n";
  bool IsMatch = Match(Seq1[*iPtr-1],Seq2[*jPtr-1]);

  auto Score = Scoring.getMatchProfit();
  if (!IsMatch) Score = Scoring.getMismatchPenalty();

  ScoreSystemType Seq1GapOpeningPenalty, Seq2GapOpeningPenalty;
  ScoreSystemType Seq1GapExtendingPenalty, Seq2GapExtendingPenalty;

  Seq1GapOpeningPenalty = Seq2GapOpeningPenalty = Scoring.getGapStartPenalty() + Scoring.getGapExtendPenalty();
  Seq1GapExtendingPenalty = Seq2GapExtendingPenalty = Scoring.getGapExtendPenalty();

  // Free gaps at the ends
  if(!Scoring.getPenalizeEndingGap()) {
    if(*iPtr == (ScoreHeight-1)) Seq1GapOpeningPenalty = Seq1GapExtendingPenalty = 0;
    if(*jPtr == (ScoreWidth-1)) Seq2GapOpeningPenalty = Seq2GapExtendingPenalty = 0;
  }
  if(!Scoring.getPenalizeStartingGap()) {
    if(*iPtr == 0) Seq1GapOpeningPenalty = Seq1GapExtendingPenalty = 0;
    if(*jPtr == 0) Seq2GapOpeningPenalty = Seq2GapExtendingPenalty = 0;
  }

  ScoreSystemType PreviousMatchPenalty, PreviousSeq1GapPenalty, PreviousSeq2GapPenalty;

  switch(*CurrMatrixPtr) {
    case MK_MATCH:
      PreviousMatchPenalty = Score;
      PreviousSeq1GapPenalty = Score;
      PreviousSeq2GapPenalty = Score;
      (*iPtr)--;
      (*jPtr)--;
      break;

    case MK_GAP_1:
      PreviousMatchPenalty = Seq2GapOpeningPenalty;
      PreviousSeq1GapPenalty = Seq2GapOpeningPenalty;
      PreviousSeq2GapPenalty = Seq2GapExtendingPenalty;
      (*jPtr)--;
      break;

    case MK_GAP_2:
      PreviousMatchPenalty = Seq1GapOpeningPenalty;
      PreviousSeq1GapPenalty = Seq1GapExtendingPenalty;
      PreviousSeq2GapPenalty = Seq1GapOpeningPenalty;
      (*iPtr)--;
      break;

    default:
      assert(false && "Should NOT be here!");
  }

  if((Scoring.getGapRestriction()!=ScoringSystem::GapRestrictionKind::GRK_NO_GAP_IN_SEQ1
     || *iPtr == 0 || *iPtr == (ScoreHeight-1)) &&
     GapIn2Matrix[(*iPtr)*ScoreWidth + (*jPtr)] + PreviousSeq1GapPenalty == *CurrScorePtr) {
    *CurrMatrixPtr = MK_GAP_2;
    *CurrScorePtr = GapIn2Matrix[(*iPtr)*ScoreWidth + (*jPtr)];
  } else if((Scoring.getGapRestriction()!=ScoringSystem::GapRestrictionKind::GRK_NO_GAP_IN_SEQ2
     || *jPtr == 0 || *jPtr == (ScoreWidth-1)) &&
     GapIn1Matrix[(*iPtr)*ScoreWidth + (*jPtr)] + PreviousSeq2GapPenalty == *CurrScorePtr) {
    *CurrMatrixPtr = MK_GAP_1;
    *CurrScorePtr = GapIn1Matrix[(*iPtr)*ScoreWidth + (*jPtr)];
  } else if(MatchMatrix[(*iPtr)*ScoreWidth + (*jPtr)] + PreviousMatchPenalty == *CurrScorePtr) {
    *CurrMatrixPtr = MK_MATCH;
    *CurrScorePtr = MatchMatrix[(*iPtr)*ScoreWidth + (*jPtr)];
  } else {
    assert(false && "Should NOT be here!");
  }
}


#ifdef NOT_IN_LLVM
#else
} // namespace
#endif

#endif
