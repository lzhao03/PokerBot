#include "src/betting_abstraction.h"

#include <algorithm>
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

void AddAction(std::array<GameAction, GameTree::kMaxActionsPerNode>& actions,
               uint8_t& action_count,
               ActionKind kind,
               int amount) {
  if (action_count >= GameTree::kMaxActionsPerNode) {
    throw std::logic_error("Legal action table exceeded kMaxActionsPerNode");
  }
  actions[static_cast<size_t>(action_count)] = {kind, amount, -1};
  ++action_count;
}

void AddActionIfMissing(
    std::array<GameAction, GameTree::kMaxActionsPerNode>& actions,
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

uint8_t BettingAbstraction::actions_for_betting_node(
    const CompactPublicState& state,
    int player,
    std::array<GameAction, GameTree::kMaxActionsPerNode>& actions) const {
  uint8_t action_count = 0;
  const int stack = state.stack[player];
  if (stack <= 0) {
    return action_count;
  }

  const int to_call =
      std::max(0, state.player_contribution[Opponent(player)] -
                      state.player_contribution[player]);
  if (to_call > 0) {
    AddAction(actions, action_count, ActionKind::kFold, 0);
    AddAction(actions, action_count, ActionKind::kCall,
              std::min(to_call, stack));
  } else {
    AddAction(actions, action_count, ActionKind::kCheck, 0);
  }

  const ActionKind sized_action =
      to_call > 0 ? ActionKind::kRaise : ActionKind::kBet;
  for (double bet_size : BetSizesForStreet(config_, state.street)) {
    const int amount = to_call + ConcreteBetAmount(state, bet_size);
    if (amount < stack) {
      AddActionIfMissing(actions, action_count, sized_action, amount);
    }
  }

  if (to_call == 0 || stack > to_call) {
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

}  // namespace poker
