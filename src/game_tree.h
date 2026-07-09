#pragma once

#include <cstdint>
#include <limits>
#include "absl/types/span.h"
#include "src/hand_evaluator.h"
#include "src/poker_types.h"

namespace poker {

class GameTree {
public:
  // Maximum number of legal actions at any decision node.
  // With 3 bet sizes: fold/call + 3 raises + all-in = 6 (facing bet),
  // or check + 3 bets + all-in = 5 (no bet). 8 gives headroom.
  static constexpr int kMaxActionsPerNode = 8;
  static constexpr uint32_t kInvalidPublicStateId =
      std::numeric_limits<uint32_t>::max();
  static constexpr uint32_t kInvalidBettingHistoryId =
      std::numeric_limits<uint32_t>::max();
  
  GameTree() = default;
  
  // Apply an action to a state to get the next state
  CompactPublicState apply_action(const CompactPublicState& state,
                                  const GameAction& action) const;

  // Apply sampled public cards at a chance node to get the next state.
  CompactPublicState apply_chance(
      const CompactPublicState& state,
      absl::Span<const CardId> cards) const;
  
  // Get the utility at a terminal state
  double get_utility(const CompactPublicState& state, ComboId player_a_hand,
                     ComboId player_b_hand) const;
  
  // Check if a state is terminal
  bool is_terminal(const CompactPublicState& state) const;
  
  // Get the player to act at a given state
  int get_player_to_act(const CompactPublicState& state) const;
  
  // Check if a betting round is over
  bool is_betting_round_over(const CompactPublicState& state) const;
  
private:
  HandEvaluator hand_evaluator_;
};

} // namespace poker
