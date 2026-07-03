#ifndef POKER_SUBGAME_VALUE_H_
#define POKER_SUBGAME_VALUE_H_

#include "src/continuation_value.h"
#include "src/poker.pb.h"

namespace poker {

class NestedCFRContinuationValueProvider : public ContinuationValueProvider {
 public:
  NestedCFRContinuationValueProvider(PokerConfig config, int iterations);

  double value(GameTree* game_tree,
               const BoardState& state,
               const Hand& player_a_hand,
               const Hand& player_b_hand) const override;

 private:
  PokerConfig config_;
  int iterations_;
};

}  // namespace poker

#endif  // POKER_SUBGAME_VALUE_H_
