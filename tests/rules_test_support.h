#pragma once

#include "src/game_state.h"

namespace poker::test {

inline ExactGameState InitialHeadsUpState(Chips player0_stack,
                                          Chips player1_stack,
                                          Chips small_blind,
                                          Chips big_blind) {
  ExactGameState state;
  state.betting.stack = {
      player0_stack - small_blind,
      player1_stack - big_blind,
  };
  state.betting.committed = {small_blind, big_blind};
  return state;
}

}  // namespace poker::test
