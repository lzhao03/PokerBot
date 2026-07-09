#include "src/betting_abstraction.h"

#include <algorithm>
#include <stdexcept>

namespace poker {
namespace {

constexpr int kActionKeyMultiplier = 1000000;

int Stack(const CompactPublicState& state, int player) {
  return state.stack[player];
}

int Contribution(const CompactPublicState& state, int player) {
  return state.player_contribution[player];
}

int OutstandingToCall(const CompactPublicState& state, int player) {
  return std::max(0, Contribution(state, Opponent(player)) -
                         Contribution(state, player));
}

int StreetBetSizesSize(const SolverConfig& config, StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return config.preflop_bet_sizes.size();
    case StreetKind::kFlop:
      return config.flop_bet_sizes.size();
    case StreetKind::kTurn:
      return config.turn_bet_sizes.size();
    case StreetKind::kRiver:
      return config.river_bet_sizes.size();
  }
}

double BetSizeForStreet(const SolverConfig& config,
                        StreetKind street,
                        int index) {
  if (StreetBetSizesSize(config, street) == 0) {
    return config.bet_sizes[index];
  }

  switch (street) {
    case StreetKind::kPreflop:
      return config.preflop_bet_sizes[index];
    case StreetKind::kFlop:
      return config.flop_bet_sizes[index];
    case StreetKind::kTurn:
      return config.turn_bet_sizes[index];
    case StreetKind::kRiver:
      return config.river_bet_sizes[index];
  }
}

int BetSizesSize(const SolverConfig& config, StreetKind street) {
  const int street_sizes = StreetBetSizesSize(config, street);
  return street_sizes > 0 ? street_sizes
                          : static_cast<int>(config.bet_sizes.size());
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
  const int stack = Stack(state, player);
  if (stack <= 0) {
    return action_count;
  }

  const int to_call = OutstandingToCall(state, player);
  if (to_call > 0) {
    AddAction(actions, action_count, ActionKind::kFold, 0);
    AddAction(actions, action_count, ActionKind::kCall,
              std::min(to_call, stack));

    for (int i = 0; i < BetSizesSize(config_, state.street); ++i) {
      const int raise_amount =
          to_call + ConcreteBetAmount(
                        state, BetSizeForStreet(config_, state.street, i));
      if (raise_amount < stack) {
        AddActionIfMissing(actions, action_count, ActionKind::kRaise,
                           raise_amount);
      }
    }
    if (stack > to_call) {
      AddAction(actions, action_count, ActionKind::kAllIn, stack);
    }
  } else {
    AddAction(actions, action_count, ActionKind::kCheck, 0);

    for (int i = 0; i < BetSizesSize(config_, state.street); ++i) {
      const int bet_amount =
          ConcreteBetAmount(state, BetSizeForStreet(config_, state.street, i));
      if (bet_amount < stack) {
        AddActionIfMissing(actions, action_count, ActionKind::kBet,
                           bet_amount);
      }
    }
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
