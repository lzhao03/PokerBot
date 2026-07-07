#ifndef POKER_TERMINAL_UTILITY_CACHE_H_
#define POKER_TERMINAL_UTILITY_CACHE_H_

#include <cstdint>
#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>

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
    size_t hash = 0;
    int street = 0;
    int pot = 0;
    int player_a_contribution = 0;
    int player_b_contribution = 0;
    int board_size = 0;
    ComboId player_a_hand = 0;
    ComboId player_b_hand = 0;
    CardMask board_mask = 0;

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
  double get_or_compute(const CompactPublicState& state,
                        ComboId player_a_hand,
                        ComboId player_b_hand,
                        const std::function<double()>& compute);

  Stats stats() const;

 private:
  static Key key_for(const CompactPublicState& state,
                     ComboId player_a_hand,
                     ComboId player_b_hand);
  double get_or_compute_key(Key key, const std::function<double()>& compute);
  static size_t compute_hash(const Key& key);

  mutable std::array<Shard, kShardCount> shards_;
};

}  // namespace poker

#endif  // POKER_TERMINAL_UTILITY_CACHE_H_
