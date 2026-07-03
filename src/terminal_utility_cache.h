#ifndef POKER_TERMINAL_UTILITY_CACHE_H_
#define POKER_TERMINAL_UTILITY_CACHE_H_

#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>

#include "src/poker.pb.h"

namespace poker {

class TerminalUtilityCache {
 public:
  struct Stats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  double get_or_compute(const BoardState& state,
                        const Hand& player_a_hand,
                        const Hand& player_b_hand,
                        const std::function<double()>& compute);

  Stats stats() const;

 private:
  struct Key {
    int street = 0;
    int pot = 0;
    int player_a_contribution = 0;
    int player_b_contribution = 0;
    int board_size = 0;
    std::array<int, 2> player_a_cards = {-1, -1};
    std::array<int, 2> player_b_cards = {-1, -1};
    std::array<int, 5> board_cards = {-1, -1, -1, -1, -1};

    bool operator==(const Key& other) const;
  };

  struct KeyHash {
    size_t operator()(const Key& key) const;
  };

  static Key key_for(const BoardState& state,
                     const Hand& player_a_hand,
                     const Hand& player_b_hand);

  mutable std::mutex mutex_;
  std::unordered_map<Key, double, KeyHash> values_;
  int64_t hits_ = 0;
  int64_t misses_ = 0;
};

}  // namespace poker

#endif  // POKER_TERMINAL_UTILITY_CACHE_H_
