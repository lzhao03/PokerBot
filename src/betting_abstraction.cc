#include "src/betting_abstraction.h"

#include <vector>

namespace poker {
namespace {

const std::vector<double>& BetSizesForStreet(const SolverConfig& config,
                                             StreetKind street) {
  return config.bet_sizes[static_cast<size_t>(street)];
}

}  // namespace

BettingAbstraction::BettingAbstraction(const SolverConfig& config)
    : config_(config) {}

ActionMenu BettingAbstraction::actions_for_betting_node(
    const BettingState& state) const {
  return LegalActions(state, BetSizesForStreet(config_, state.street));
}

}  // namespace poker
