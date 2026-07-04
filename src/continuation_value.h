#ifndef POKER_CONTINUATION_VALUE_H_
#define POKER_CONTINUATION_VALUE_H_

#include "src/poker.pb.h"
#include "src/training_range.h"

namespace poker {

class GameTree;

struct ContinuationContext {
  BoardState state;
  Hand player_a_hand;
  Hand player_b_hand;
  TrainingRangeView player_a_range;
  TrainingRangeView player_b_range;

  static ContinuationContext ExactHands(const BoardState& state,
                                        const Hand& player_a_hand,
                                        const Hand& player_b_hand);
  bool has_ranges() const;
};

class ContinuationValueProvider {
 public:
  virtual ~ContinuationValueProvider() = default;

  double value(GameTree& game_tree,
               const BoardState& state,
               const Hand& player_a_hand,
               const Hand& player_b_hand) const;
  virtual double value(GameTree& game_tree,
                       const ContinuationContext& context) const = 0;
};

class BettingRoundTerminalValueProvider : public ContinuationValueProvider {
 public:
  using ContinuationValueProvider::value;

  double value(GameTree& game_tree,
               const ContinuationContext& context) const override;
};

}  // namespace poker

#endif  // POKER_CONTINUATION_VALUE_H_
