#ifndef POKER_TERMINAL_UTILITY_CACHE_H_
#define POKER_TERMINAL_UTILITY_CACHE_H_

#include <array>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "src/combo.h"
#include "src/poker_types.h"

namespace poker {

class TerminalUtilityCache {
 public:
  struct Stats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  // ponytail: revisit this cache once terminal evaluation is cheap; cap or
  // bypass low-reuse results instead of spending memory on one-off showdowns.

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

  struct Shard {
    mutable std::mutex mutex;
    std::unordered_map<Key, double, KeyHash> values;
    int64_t hits = 0;
    int64_t misses = 0;
  };

  static constexpr size_t kShardCount = 64;

 public:
  template <typename State, typename Compute>
  double get_or_compute(const State& state,
                        ComboId player_a_hand,
                        ComboId player_b_hand,
                        Compute compute) {
    Key key = key_for(state, player_a_hand, player_b_hand);
    Shard& shard = shards_[KeyHash{}(key) & (kShardCount - 1)];

    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      auto it = shard.values.find(key);
      if (it != shard.values.end()) {
        ++shard.hits;
        return it->second;
      }
    }

    double value = compute();

    {
      std::lock_guard<std::mutex> lock(shard.mutex);
      auto inserted = shard.values.emplace(std::move(key), value);
      if (inserted.second) {
        ++shard.misses;
        return value;
      }
      ++shard.hits;
      return inserted.first->second;
    }
  }

  Stats stats() const;

 private:
  static Key key_for(const GameState& state,
                     ComboId player_a_hand,
                     ComboId player_b_hand);
  static Key key_for(const CompactPublicState& state,
                     ComboId player_a_hand,
                     ComboId player_b_hand);

  mutable std::array<Shard, kShardCount> shards_;
};

}  // namespace poker

#endif  // POKER_TERMINAL_UTILITY_CACHE_H_
