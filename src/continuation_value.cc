#include "src/continuation_value.h"

#include "src/game_tree.h"

namespace poker {

double BettingRoundTerminalValueProvider::value(
    GameTree* game_tree,
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand) const {
  if (game_tree->is_betting_round_over(state)) {
    return game_tree->get_utility(state, player_a_hand, player_b_hand);
  }
  return 0.0;
}

}  // namespace poker
