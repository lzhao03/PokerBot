#include "src/continuation_value.h"

#include "src/game_tree.h"

namespace poker {

ContinuationContext ContinuationContext::ExactHands(
    const GameState& state,
    ComboId player_a_hand,
    ComboId player_b_hand) {
  ContinuationContext context;
  context.state = state;
  context.player_a_hand = player_a_hand;
  context.player_b_hand = player_b_hand;
  return context;
}

bool ContinuationContext::has_ranges() const {
  return !player_a_range.empty() && !player_b_range.empty();
}

double ContinuationValueProvider::value(
    GameTree& game_tree,
    const GameState& state,
    ComboId player_a_hand,
    ComboId player_b_hand) const {
  return value(game_tree,
               ContinuationContext::ExactHands(
                   state, player_a_hand, player_b_hand));
}

double BettingRoundTerminalValueProvider::value(
    GameTree& game_tree,
    const ContinuationContext& context) const {
  if (game_tree.is_betting_round_over(context.state)) {
    return game_tree.get_utility(
        context.state, context.player_a_hand, context.player_b_hand);
  }
  return 0.0;
}

}  // namespace poker
