#ifndef POKER_SUBGAME_VALUE_H_
#define POKER_SUBGAME_VALUE_H_

#include "src/continuation_value.h"
#include "src/poker.pb.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace poker {

// Solves a cutoff state for one exact private-hand assignment.
// This is useful as continuation-value plumbing, but it is not range-aware
// poker subgame solving.
class ExactHandNestedCFRContinuationValueProvider
    : public ContinuationValueProvider {
 public:
  struct Stats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  ExactHandNestedCFRContinuationValueProvider(PokerConfig config,
                                              int iterations);

  double value(GameTree* game_tree,
               const BoardState& state,
               const Hand& player_a_hand,
               const Hand& player_b_hand) const override;
  Stats stats() const;

 private:
  double compute_value(GameTree* game_tree,
                       const BoardState& state,
                       const Hand& player_a_hand,
                       const Hand& player_b_hand) const;

  PokerConfig config_;
  int iterations_;
  mutable std::mutex mutex_;
  mutable std::unordered_map<std::string, double> values_;
  mutable int64_t hits_ = 0;
  mutable int64_t misses_ = 0;
};

}  // namespace poker

#endif  // POKER_SUBGAME_VALUE_H_
