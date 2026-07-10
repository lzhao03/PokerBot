#pragma once

#include "src/game_rules.h"
#include "src/poker_types.h"

namespace poker {

class BettingAbstraction {
 public:
  explicit BettingAbstraction(const SolverConfig& config);

  ActionMenu actions_for_betting_node(const BettingState& state) const;

 private:
  SolverConfig config_;
};

}  // namespace poker
