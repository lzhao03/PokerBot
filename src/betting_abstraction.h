#pragma once

#include <algorithm>
#include <cstddef>

#include "src/build_flags.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"

namespace poker {

class BettingAbstraction {
 public:
  using BettingHistoryKey = StrategyTables::BettingHistoryKey;
  using BettingHistoryRow = StrategyTables::BettingHistoryRow;

  BettingHistoryKey make_history_key(const CompactPublicState& state) const {
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

    const int history_value_count = state.history_size * 3;
    if (history_value_count > BettingHistoryKey::kInlineHistoryValues) {
      key.history_overflow.reserve(history_value_count -
                                   BettingHistoryKey::kInlineHistoryValues);
    }
    for (uint16_t i = 0; i < state.history_size; ++i) {
      const CompactAction action = CompactHistoryAction(state, i);
      AddHistoryValue(key, action.player);
      AddHistoryValue(key, static_cast<int>(action.kind));
      AddHistoryValue(key, action.amount);
    }

    return key;
  }

  BettingHistoryRow make_history_row(const CompactPublicState& state) const {
    BettingHistoryRow row;
    row.street = static_cast<int>(state.street);
    row.pot = state.pot;
    row.stack = state.stack;
    row.all_in = state.all_in ? 1 : 0;
    row.folded_player = state.folded_player;
    row.player_to_act = state.player_to_act;
    row.player_contributions = state.player_contribution;
    ApplyProjection(state, row);
    return row;
  }

  CompactPublicState public_state_for_row(CompactPublicState state) const {
    ApplyProjection(state);
    return state;
  }

  void copy_history_to_row(const BettingHistoryKey& key,
                           BettingHistoryRow& row) const {
    row.history_size = key.history_size;
    row.history_values = key.history_values;
    row.history_overflow = key.history_overflow;
  }

  void replace_with_action_index_history(
      const BettingHistoryRow& parent_row,
      int action_index,
      const CompactPublicState& child_state,
      BettingHistoryKey& key) const {
    key.history_size = 0;
    key.history_overflow.clear();
    for (int i = 0; i < parent_row.history_size; ++i) {
      AddHistoryValue(key, RowHistoryValue(parent_row, i));
    }
    const CompactAction action = child_state.last_action;
    AddHistoryValue(key, action.player);
    AddHistoryValue(key, static_cast<int>(action.kind));
    AddHistoryValue(key, action_index);
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

  static int BucketChips(int chips) {
    if (chips <= 0) {
      return 0;
    }
    int bucket = 1;
    while (chips > 1) {
      chips >>= 1;
      ++bucket;
    }
    return bucket;
  }

  static Projection Project(const CompactPublicState& state) {
    const int contribution_gap =
        state.player_contribution[0] > state.player_contribution[1]
            ? state.player_contribution[0] - state.player_contribution[1]
            : state.player_contribution[1] - state.player_contribution[0];
    return {
        static_cast<int>(state.street),
        BucketChips(state.pot),
        BucketChips(std::min(state.stack[0], state.stack[1])),
        BucketChips(contribution_gap),
        state.all_in ? 1 : 0,
        state.folded_player,
        state.player_to_act,
    };
  }

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

  static void ApplyProjection(const CompactPublicState& state,
                              BettingHistoryRow& row) {
    if constexpr (kCoarsePublicBuckets) {
      const Projection projection = Project(state);
      row.street = projection.street;
      row.pot = projection.pot_bucket;
      row.stack = {projection.effective_stack_bucket, 0};
      row.all_in = projection.all_in;
      row.folded_player = projection.folded_player;
      row.player_to_act = projection.player_to_act;
      row.player_contributions = {projection.to_call_bucket, 0};
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

  static void AddHistoryValue(BettingHistoryKey& key, int value) {
    if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
      key.history_values[static_cast<size_t>(key.history_size)] = value;
    } else {
      key.history_overflow.push_back(value);
    }
    ++key.history_size;
  }

  static int RowHistoryValue(const BettingHistoryRow& row, int index) {
    if (index < BettingHistoryKey::kInlineHistoryValues) {
      return row.history_values[static_cast<size_t>(index)];
    }
    return row.history_overflow[
        static_cast<size_t>(index - BettingHistoryKey::kInlineHistoryValues)];
  }
};

}  // namespace poker
