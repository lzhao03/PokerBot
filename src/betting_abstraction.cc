#include "src/betting_abstraction.h"

#include <vector>

namespace poker {
namespace {

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

}  // namespace

BettingAbstraction::BettingAbstraction(const SolverConfig& config)
    : config_(config) {}

ActionMenu BettingAbstraction::actions_for_betting_node(
    const BettingState& state) const {
  return LegalActions(state, BetSizesForStreet(config_, state.street));
}

}  // namespace poker
