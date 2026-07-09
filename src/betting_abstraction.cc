#include "src/betting_abstraction.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

constexpr int kActionKeyMultiplier = 1000000;

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

int ConcreteBetAmount(const CompactPublicState& state, double size) {
  if (size <= 0.0) {
    return 0;
  }
  return std::max(1, static_cast<int>(std::max(1, state.pot) * size));
}

void AddAction(std::array<GameAction, kMaxActionsPerNode>& actions,
               uint8_t& action_count,
               ActionKind kind,
               int amount) {
  if (action_count >= kMaxActionsPerNode) {
    throw std::logic_error("Legal action table exceeded kMaxActionsPerNode");
  }
  actions[static_cast<size_t>(action_count)] = {kind, amount, -1};
  ++action_count;
}

void AddActionIfMissing(
    std::array<GameAction, kMaxActionsPerNode>& actions,
    uint8_t& action_count,
    ActionKind kind,
    int amount) {
  for (uint8_t i = 0; i < action_count; ++i) {
    const GameAction& action = actions[static_cast<size_t>(i)];
    if (action.kind == kind && action.amount == amount) {
      return;
    }
  }
  AddAction(actions, action_count, kind, amount);
}

}  // namespace

BettingAbstraction::BettingAbstraction(const SolverConfig& config)
    : config_(config) {}

int BettingAbstraction::BucketChips(int chips) {
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

BettingAbstraction::Projection BettingAbstraction::Project(
    const CompactPublicState& state) {
  const int gap =
      state.player_contribution[0] > state.player_contribution[1]
          ? state.player_contribution[0] - state.player_contribution[1]
          : state.player_contribution[1] - state.player_contribution[0];
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

uint8_t BettingAbstraction::actions_for_betting_node(
    const CompactPublicState& state,
    int player,
    std::array<GameAction, kMaxActionsPerNode>& actions) const {
  uint8_t action_count = 0;
  const int stack = state.stack[player];
  if (stack <= 0) {
    return action_count;
  }

  const int opponent_chips = state.player_contribution[Opponent(player)];
  const int player_chips = state.player_contribution[player];
  const int outstanding_call = std::max(0, opponent_chips - player_chips);
  const bool facing_bet = outstanding_call > 0;
  if (facing_bet) {
    const int call_amount = std::min(outstanding_call, stack);
    AddAction(actions, action_count, ActionKind::kFold, 0);
    AddAction(actions, action_count, ActionKind::kCall, call_amount);
  } else {
    AddAction(actions, action_count, ActionKind::kCheck, 0);
  }

  const ActionKind sized_action = facing_bet ? ActionKind::kRaise : ActionKind::kBet;
  for (double bet_size : BetSizesForStreet(config_, state.street)) {
    const int amount = outstanding_call + ConcreteBetAmount(state, bet_size);
    if (amount < stack) {
      AddActionIfMissing(actions, action_count, sized_action, amount);
    }
  }

  if (!facing_bet || stack > outstanding_call) {
    AddAction(actions, action_count, ActionKind::kAllIn, stack);
  }

  return action_count;
}

int BettingAbstraction::action_key(const GameAction& action) const {
  if (action.amount < 0 || action.amount >= kActionKeyMultiplier) {
    throw std::invalid_argument("Action amount is outside action-key range");
  }
  return static_cast<int>(action.kind) * kActionKeyMultiplier + action.amount;
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

void BettingAbstraction::AppendHistoryValue(BettingHistoryKey& key,
                                            int value) {
  if (key.history_size < BettingHistoryKey::kInlineHistoryValues) {
    key.history_values[static_cast<size_t>(key.history_size)] = value;
  } else {
    key.history_overflow.push_back(value);
  }
  ++key.history_size;
}

void BettingAbstraction::AppendStateHistory(const CompactPublicState& state,
                                            BettingHistoryKey& key) {
  const int value_count = state.history_size * 3;
  if (value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  for (uint16_t i = 0; i < state.history_size; ++i) {
    const CompactAction action = CompactHistoryAction(state, i);
    AppendHistoryValue(key, action.player);
    AppendHistoryValue(key, static_cast<int>(action.kind));
    AppendHistoryValue(key, action.amount);
  }
}

void BettingAbstraction::AppendRowHistory(const BettingHistoryRow& row,
                                          BettingHistoryKey& key) {
  const int value_count = row.history_size + 3;
  if (value_count > BettingHistoryKey::kInlineHistoryValues) {
    key.history_overflow.reserve(value_count -
                                 BettingHistoryKey::kInlineHistoryValues);
  }
  for (int i = 0; i < row.history_size; ++i) {
    AppendHistoryValue(key, RowHistoryValue(row, i));
  }
}

int BettingAbstraction::RowHistoryValue(const BettingHistoryRow& row,
                                        int index) {
  if (index < BettingHistoryKey::kInlineHistoryValues) {
    return row.history_values[static_cast<size_t>(index)];
  }
  return row.history_overflow[
      static_cast<size_t>(index - BettingHistoryKey::kInlineHistoryValues)];
}

}  // namespace poker
