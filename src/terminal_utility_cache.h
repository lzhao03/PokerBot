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
    size_t hash = 0;
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
    return get_or_compute_key(std::move(key), compute);
  }

  template <typename Compute>
  double get_or_compute(StreetKind street,
                        int pot,
                        int player_a_contribution,
                        int player_b_contribution,
                        const std::array<CardId, kMaxBoardCards>& board_cards,
                        int board_count,
                        ComboId player_a_hand,
                        ComboId player_b_hand,
                        Compute compute) {
    Key key = key_for(street, pot, player_a_contribution,
                      player_b_contribution, board_cards, board_count,
                      player_a_hand, player_b_hand);
    return get_or_compute_key(std::move(key), compute);
  }

  Stats stats() const;

 private:
  template <typename Compute>
  double get_or_compute_key(Key key, Compute compute) {
    Shard& shard = shards_[key.hash & (kShardCount - 1)];

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

  static Key key_for(const GameState& state,
                     ComboId player_a_hand,
                     ComboId player_b_hand);
  static Key key_for(const CompactPublicState& state,
                     ComboId player_a_hand,
                     ComboId player_b_hand);
  static Key key_for(StreetKind street,
                     int pot,
                     int player_a_contribution,
                     int player_b_contribution,
                     const std::array<CardId, kMaxBoardCards>& board_cards,
                     int board_count,
                     ComboId player_a_hand,
                     ComboId player_b_hand);
  static size_t compute_hash(const Key& key);

  mutable std::array<Shard, kShardCount> shards_;
};

}  // namespace poker

#endif  // POKER_TERMINAL_UTILITY_CACHE_H_
