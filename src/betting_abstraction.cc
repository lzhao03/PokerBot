#include "src/betting_abstraction.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

constexpr int kActionKeyMultiplier = 1000000;

using ActionMenu = BettingAbstraction::ActionMenu;

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

}  // namespace poker
