#pragma once

#include "src/poker.h"

namespace poker::test {

inline ExactPublicState InitialHeadsUpState(Chips player0_stack,
                                           Chips player1_stack,
                                           Chips small_blind,
                                           Chips big_blind) {
  return MakeInitialState(BettingRules{big_blind},
                          {player0_stack, player1_stack},
                          {small_blind, big_blind});
}

}  // namespace poker::test
