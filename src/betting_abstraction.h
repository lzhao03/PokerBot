#pragma once

#include <array>

#include "src/poker_types.h"

namespace poker {

class BettingAbstraction {
 public:
  struct ActionMenu {
    uint8_t count = 0;
    std::array<GameAction, kMaxActionsPerNode> actions = {};
  };

  explicit BettingAbstraction(const SolverConfig& config);

  ActionMenu actions_for_betting_node(const BettingState& state,
                                      int player) const;

 private:
  SolverConfig config_;
};

}  // namespace poker
