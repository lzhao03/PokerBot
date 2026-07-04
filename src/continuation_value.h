#ifndef POKER_CONTINUATION_VALUE_H_
#define POKER_CONTINUATION_VALUE_H_

#include "src/combo.h"
#include "src/poker_types.h"
#include "src/training_range.h"

namespace poker {

class GameTree;

struct ContinuationContext {
  GameState state;
  ComboId player_a_hand = 0;
  ComboId player_b_hand = 0;
  TrainingRangeView player_a_range;
  TrainingRangeView player_b_range;

  static ContinuationContext ExactHands(const GameState& state,
                                        ComboId player_a_hand,
                                        ComboId player_b_hand);
  bool has_ranges() const;
};

class ContinuationValueProvider {
 public:
  virtual ~ContinuationValueProvider() = default;

  double value(GameTree& game_tree,
               const GameState& state,
               ComboId player_a_hand,
               ComboId player_b_hand) const;
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
