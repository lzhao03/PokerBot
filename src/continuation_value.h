#ifndef POKER_CONTINUATION_VALUE_H_
#define POKER_CONTINUATION_VALUE_H_

#include "src/poker.pb.h"

namespace poker {

class GameTree;

class ContinuationValueProvider {
 public:
  virtual ~ContinuationValueProvider() = default;

  virtual double value(GameTree* game_tree,
                       const BoardState& state,
                       const Hand& player_a_hand,
                       const Hand& player_b_hand) const = 0;
};

class BettingRoundTerminalValueProvider : public ContinuationValueProvider {
 public:
  double value(GameTree* game_tree,
               const BoardState& state,
               const Hand& player_a_hand,
               const Hand& player_b_hand) const override;
};

}  // namespace poker

#endif  // POKER_CONTINUATION_VALUE_H_
