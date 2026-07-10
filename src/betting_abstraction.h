#pragma once

#include <array>

#include "src/poker_types.h"
#include "src/strategy_tables.h"

namespace poker {

class BettingAbstraction {
 public:
  using BettingHistoryKey = StrategyTables::BettingHistoryKey;
  using BettingHistoryRow = StrategyTables::BettingHistoryRow;

  struct ActionMenu {
    uint8_t count = 0;
    std::array<GameAction, kMaxActionsPerNode> actions = {};
  };

  explicit BettingAbstraction(const SolverConfig& config);

  ActionMenu actions_for_betting_node(const BettingState& state,
                                      int player) const;

  int action_key(const GameAction& action) const;

  BettingHistoryKey make_history_key(const BettingState& state) const;

  BettingHistoryKey make_action_child_history_key(
      const BettingHistoryRow& parent_row,
      int action_index,
      const BettingState& child_state) const;

  BettingHistoryRow make_history_row(const BettingHistoryKey& key) const;

 private:
  SolverConfig config_;
};

}  // namespace poker
