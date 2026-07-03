#include "src/subgame_value.h"

#include "src/cfr_solver.h"
#include "src/game_tree.h"

#include <stdexcept>
#include <utility>

namespace poker {

NestedCFRContinuationValueProvider::NestedCFRContinuationValueProvider(
    PokerConfig config,
    int iterations)
    : config_(std::move(config)), iterations_(iterations) {
  if (iterations_ <= 0) {
    throw std::invalid_argument("Subgame CFR iterations must be positive");
  }
}

double NestedCFRContinuationValueProvider::value(
    GameTree* game_tree,
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand) const {
  if (game_tree->is_terminal(state)) {
    return game_tree->get_utility(state, player_a_hand, player_b_hand);
  }

  PokerConfig subgame_config = config_;
  subgame_config.set_max_depth(0);
  CFRSolver subgame_solver(subgame_config, state);
  subgame_solver.run(iterations_, player_a_hand, player_b_hand);
  return subgame_solver.evaluate_strategy(player_a_hand, player_b_hand);
}

}  // namespace poker
