#include "src/betting_abstraction.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "src/build_flags.h"

namespace poker {
namespace {

constexpr int kActionKeyMultiplier = 1000000;

using ActionMenu = BettingAbstraction::ActionMenu;
using BettingHistoryKey = BettingAbstraction::BettingHistoryKey;
using BettingHistoryRow = BettingAbstraction::BettingHistoryRow;

struct CoarseHistoryKeyFields {
  int street = 0;
  int pot_bucket = 0;
  int effective_stack_bucket = 0;
  int to_call_bucket = 0;
  int all_in = 0;
  int folded_player = -1;
  int player_to_act = 0;
};

const std::vector<double>& BetSizesForStreet(const SolverConfig& config,
                                             StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return config.preflop_bet_sizes.empty() ? config.bet_sizes
                                              : config.preflop_bet_sizes;
    case StreetKind::kFlop:
      return config.flop_bet_sizes.empty() ? config.bet_sizes
                                           : config.flop_bet_sizes;
    case StreetKind::kTurn:
      return config.turn_bet_sizes.empty() ? config.bet_sizes
                                           : config.turn_bet_sizes;
    case StreetKind::kRiver:
      return config.river_bet_sizes.empty() ? config.bet_sizes
                                            : config.river_bet_sizes;
  }
}

int ConcreteBetAmount(const BettingState& state, double size) {
  if (size <= 0.0) {
    return 0;
  }
  return std::max(1, static_cast<int>(std::max(1, state.pot) * size));
}

void AddAction(ActionMenu& menu, ActionKind kind, int amount) {
  if (menu.count >= kMaxActionsPerNode) {
    throw std::logic_error("Legal action table exceeded kMaxActionsPerNode");
  }
  menu.actions[static_cast<size_t>(menu.count)] = {kind, amount, -1};
  ++menu.count;
}

void AddActionIfMissing(ActionMenu& menu, ActionKind kind, int amount) {
  for (uint8_t i = 0; i < menu.count; ++i) {
    const GameAction& action = menu.actions[static_cast<size_t>(i)];
    if (action.kind == kind && action.amount == amount) {
      return;
    }
  }
  AddAction(menu, kind, amount);
}

int BucketChips(int chips) {
  if (chips <= 0) {
    return 0;
  }
  return std::bit_width(static_cast<unsigned int>(chips));
}

CoarseHistoryKeyFields CoarseHistoryFields(const BettingState& state) {
  const int gap =
      state.contribution[0] > state.contribution[1]
          ? state.contribution[0] - state.contribution[1]
          : state.contribution[1] - state.contribution[0];
  return {
      static_cast<int>(state.street),
      BucketChips(state.pot),
      BucketChips(std::min(state.stack[0], state.stack[1])),
      BucketChips(gap),
      state.all_in ? 1 : 0,
      state.folded_player,
      state.player_to_act,
  };
}

BettingHistoryKey CoarsenHistoryKey(const BettingState& state,
                                    BettingHistoryKey key) {
  if constexpr (kCoarsePublicBuckets) {
    const CoarseHistoryKeyFields fields = CoarseHistoryFields(state);
    key.street = fields.street;
    key.pot = fields.pot_bucket;
    key.stack_a = fields.effective_stack_bucket;
    key.stack_b = 0;
    key.all_in = fields.all_in;
    key.folded_player = fields.folded_player;
    key.player_to_act = fields.player_to_act;
    key.player_contribution_size = 1;
    key.player_contributions = {fields.to_call_bucket, 0};
  }
  return key;
}

BettingHistoryKey BaseHistoryKey(const BettingState& state) {
  BettingHistoryKey key;
  key.street = static_cast<int>(state.street);
  key.pot = state.pot;
  key.stack_a = state.stack[0];
  key.stack_b = state.stack[1];
  key.all_in = state.all_in ? 1 : 0;
  key.folded_player = state.folded_player;
  key.player_to_act = state.player_to_act;
  key.player_contribution_size = 2;
  key.player_contributions = state.contribution;
  return CoarsenHistoryKey(state, std::move(key));
}

void AppendHistoryValue(BettingHistoryKey& key, int value) {
  if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
    key.history_values[static_cast<size_t>(key.history_size)] = value;
  } else {
    key.history_overflow.push_back(value);
  }
  ++key.history_size;
}

int RowHistoryValue(const BettingHistoryRow& row, int index) {
  if (index < BettingHistoryKey::kInlineHistoryValues) {
    return row.history_values[static_cast<size_t>(index)];
  }
  return row.history_overflow[
      static_cast<size_t>(index - BettingHistoryKey::kInlineHistoryValues)];
}

void AppendStateHistory(const BettingState& state, BettingHistoryKey& key) {
  if (state.actions_this_street == 0 ||
      state.last_action.kind == ActionKind::kNoAction) {
    return;
  }
  if (state.actions_this_street != 1) {
    throw std::logic_error(
        "BettingState does not carry full multi-action history");
  }
  AppendHistoryValue(key, state.last_action.player);
  AppendHistoryValue(key, static_cast<int>(state.last_action.kind));
  AppendHistoryValue(key, state.last_action.amount);
}

void AppendRowHistory(const BettingHistoryRow& row, BettingHistoryKey& key) {
  const int value_count = row.history_size + 3;
  if (value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  for (int i = 0; i < row.history_size; ++i) {
    AppendHistoryValue(key, RowHistoryValue(row, i));
  }
}

}  // namespace

BettingAbstraction::BettingAbstraction(const SolverConfig& config)
    : config_(config) {}

BettingAbstraction::ActionMenu BettingAbstraction::actions_for_betting_node(
    const BettingState& state,
    int player) const {
  ActionMenu menu;
  const int stack = state.stack[player];
  if (stack <= 0) {
    return menu;
  }

  const int opponent_chips = state.contribution[Opponent(player)];
  const int player_chips = state.contribution[player];
  const int outstanding_call = std::max(0, opponent_chips - player_chips);
  if (outstanding_call > 0) {
    const int call_amount = std::min(outstanding_call, stack);
    AddAction(menu, ActionKind::kFold, 0);
    AddAction(menu, ActionKind::kCall, call_amount);
  } else {
    AddAction(menu, ActionKind::kCheck, 0);
  }

  ActionKind sized_action = ActionKind::kBet;
  if (outstanding_call > 0) {
    sized_action = ActionKind::kRaise;
  }
  for (double bet_size : BetSizesForStreet(config_, state.street)) {
    const int amount = outstanding_call + ConcreteBetAmount(state, bet_size);
    if (amount < stack) {
      AddActionIfMissing(menu, sized_action, amount);
    }
  }

  if (outstanding_call == 0 || stack > outstanding_call) {
    AddAction(menu, ActionKind::kAllIn, stack);
  }

  return menu;
}

int BettingAbstraction::action_key(const GameAction& action) const {
  if (action.amount < 0 || action.amount >= kActionKeyMultiplier) {
    throw std::invalid_argument("Action amount is outside action-key range");
  }
  return static_cast<int>(action.kind) * kActionKeyMultiplier + action.amount;
}

BettingAbstraction::BettingHistoryKey BettingAbstraction::make_history_key(
    const BettingState& state) const {
  BettingHistoryKey key = BaseHistoryKey(state);
  AppendStateHistory(state, key);
  return key;
}

BettingAbstraction::BettingHistoryKey
BettingAbstraction::make_action_child_history_key(
    const BettingHistoryRow& parent_row,
    int action_index,
    const BettingState& child_state) const {
  BettingHistoryKey key = BaseHistoryKey(child_state);
  AppendRowHistory(parent_row, key);
  const GameAction action = child_state.last_action;
  AppendHistoryValue(key, action.player);
  AppendHistoryValue(key, static_cast<int>(action.kind));
  AppendHistoryValue(key, action_index);
  return key;
}

BettingAbstraction::BettingHistoryRow BettingAbstraction::make_history_row(
    const BettingHistoryKey& key) const {
  BettingHistoryRow row;
  row.street = key.street;
  row.pot = key.pot;
  row.stack = {key.stack_a, key.stack_b};
  row.all_in = key.all_in;
  row.folded_player = key.folded_player;
  row.player_to_act = key.player_to_act;
  row.player_contributions = key.player_contributions;
  row.history_size = key.history_size;
  row.history_values = key.history_values;
  row.history_overflow = key.history_overflow;
  return row;
}

}  // namespace poker
