#pragma once

#include <array>

#include "src/build_flags.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"

namespace poker {

class BettingAbstraction {
 public:
  using BettingHistoryKey = StrategyTables::BettingHistoryKey;
  using BettingHistoryRow = StrategyTables::BettingHistoryRow;

  explicit BettingAbstraction(const SolverConfig& config);

  uint8_t actions_for_betting_node(
      const CompactPublicState& state,
      int player,
      std::array<GameAction, kMaxActionsPerNode>& actions) const;

  int action_key(const GameAction& action) const;

  BettingHistoryKey make_history_key(const CompactPublicState& state) const {
    BettingHistoryKey key = BaseHistoryKey(state);
    AppendStateHistory(state, key);
    return key;
  }

  BettingHistoryKey make_action_child_history_key(
      const BettingHistoryRow& parent_row,
      int action_index,
      const CompactPublicState& child_state) const {
    BettingHistoryKey key = BaseHistoryKey(child_state);
    AppendRowHistory(parent_row, key);
    const CompactAction action = child_state.last_action;
    AppendHistoryValue(key, action.player);
    AppendHistoryValue(key, static_cast<int>(action.kind));
    AppendHistoryValue(key, action_index);
    return key;
  }

  BettingHistoryRow make_history_row(const BettingHistoryKey& key) const;

  CompactPublicState public_state_for_row(CompactPublicState state) const {
    ApplyProjection(state);
    return state;
  }

 private:
  struct Projection {
    int street = 0;
    int pot_bucket = 0;
    int effective_stack_bucket = 0;
    int to_call_bucket = 0;
    int all_in = 0;
    int folded_player = -1;
    int player_to_act = 0;
  };

  static int BucketChips(int chips);
  static Projection Project(const CompactPublicState& state);

  static void ApplyProjection(const CompactPublicState& state,
                              BettingHistoryKey& key) {
    if constexpr (kCoarsePublicBuckets) {
      const Projection projection = Project(state);
      key.street = projection.street;
      key.pot = projection.pot_bucket;
      key.stack_a = projection.effective_stack_bucket;
      key.stack_b = 0;
      key.all_in = projection.all_in;
      key.folded_player = projection.folded_player;
      key.player_to_act = projection.player_to_act;
      key.player_contribution_size = 1;
      key.player_contributions = {projection.to_call_bucket, 0};
    }
  }

  static void ApplyProjection(CompactPublicState& state) {
    if constexpr (kCoarsePublicBuckets) {
      const Projection projection = Project(state);
      state.street = static_cast<StreetKind>(projection.street);
      state.pot = projection.pot_bucket;
      state.stack = {projection.effective_stack_bucket,
                     projection.effective_stack_bucket};
      state.all_in = projection.all_in != 0;
      state.folded_player = projection.folded_player;
      state.player_to_act = projection.player_to_act;
      state.player_contribution = {0, 0};
      if (IsPlayer(projection.player_to_act)) {
        state.player_contribution[Opponent(projection.player_to_act)] =
            projection.to_call_bucket;
      }
      state.player_contribution_count = 2;
    }
  }

  static BettingHistoryKey BaseHistoryKey(const CompactPublicState& state) {
    BettingHistoryKey key;
    key.street = static_cast<int>(state.street);
    key.pot = state.pot;
    key.stack_a = state.stack[0];
    key.stack_b = state.stack[1];
    key.all_in = state.all_in ? 1 : 0;
    key.folded_player = state.folded_player;
    key.player_to_act = state.player_to_act;
    key.player_contribution_size = 2;
    key.player_contributions = state.player_contribution;
    ApplyProjection(state, key);
    return key;
  }

  static void AppendHistoryValue(BettingHistoryKey& key, int value);

  static void AppendStateHistory(const CompactPublicState& state,
                                 BettingHistoryKey& key);

  static void AppendRowHistory(const BettingHistoryRow& row,
                               BettingHistoryKey& key);

  static int RowHistoryValue(const BettingHistoryRow& row, int index);

  SolverConfig config_;
};

}  // namespace poker
