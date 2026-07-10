#pragma once

#include "src/game_state.h"

namespace poker::test {

inline ExactPublicState InitialHeadsUpState(Chips player0_stack,
                                          Chips player1_stack,
                                          Chips small_blind,
                                          Chips big_blind) {
  ExactPublicState state;
  state.betting.stack = {
      player0_stack - small_blind,
      player1_stack - big_blind,
  };
  state.betting.total_committed = {small_blind, big_blind};
  state.betting.street_committed = {small_blind, big_blind};
  state.betting.last_full_raise = big_blind;
  return state;
}

}  // namespace poker::test
