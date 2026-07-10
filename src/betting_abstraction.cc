#include "src/betting_abstraction.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "absl/container/inlined_vector.h"

namespace poker {
namespace {

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
  absl::InlinedVector<GameAction, kMaxActionsPerNode> sized_actions;
  for (double bet_size : BetSizesForStreet(config_, state.street)) {
    const int amount = outstanding_call + ConcreteBetAmount(state, bet_size);
    if (amount < stack) {
      sized_actions.push_back({sized_action, amount, -1});
    }
  }
  std::sort(sized_actions.begin(), sized_actions.end(),
            [](const GameAction& left, const GameAction& right) {
              return left.amount < right.amount;
            });
  const auto unique_end =
      std::unique(sized_actions.begin(), sized_actions.end(),
                  [](const GameAction& left, const GameAction& right) {
                    return left.kind == right.kind &&
                           left.amount == right.amount;
                  });
  sized_actions.erase(unique_end, sized_actions.end());

  for (const GameAction& action : sized_actions) {
    AddAction(menu, action.kind, action.amount);
  }

  if (outstanding_call == 0 || stack > outstanding_call) {
    AddAction(menu, ActionKind::kAllIn, stack);
  }

  return menu;
}

}  // namespace poker
