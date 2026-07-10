#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/combo.h"
#include "src/poker_types.h"

namespace poker {

inline constexpr uint32_t kInvalidPublicStateId =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kInvalidBettingHistoryId =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kCappedPublicStateId = kInvalidPublicStateId - 1;

inline constexpr size_t kCacheLineBytes = 64;
inline constexpr size_t kCumulativeActionBlockAlignment =
    kCacheLineBytes / sizeof(float);

template <typename T>
class CacheLineAlignedAllocator {
 public:
  using value_type = T;

  CacheLineAlignedAllocator() noexcept = default;
  template <typename U>
  CacheLineAlignedAllocator(const CacheLineAlignedAllocator<U>&) noexcept {}

  T* allocate(size_t n) {
    if (n > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::bad_array_new_length();
    }
    return static_cast<T*>(
        ::operator new(n * sizeof(T), std::align_val_t{kCacheLineBytes}));
  }

  void deallocate(T* ptr, size_t) noexcept {
    ::operator delete(ptr, std::align_val_t{kCacheLineBytes});
  }
};

template <typename T, typename U>
bool operator==(const CacheLineAlignedAllocator<T>&,
                const CacheLineAlignedAllocator<U>&) noexcept {
  return true;
}

template <typename T, typename U>
bool operator!=(const CacheLineAlignedAllocator<T>&,
                const CacheLineAlignedAllocator<U>&) noexcept {
  return false;
}

class StrategyTables {
 public:
  using PrivateBucketId = uint16_t;
  using PublicBucketId = uint64_t;
  static constexpr uint32_t kInvalidActionOffset =
      std::numeric_limits<uint32_t>::max();

  struct BettingHistoryKey {
    static constexpr int kInlineHistoryValues = 48;

    int street = 0;
    int pot = 0;
    int stack_a = 0;
    int stack_b = 0;
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    int player_contribution_size = 0;
    std::array<int, 2> player_contributions = {0, 0};
    int history_size = 0;
    // Only entries [0, history_size) are read by hash/equality.
    std::array<int, kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;

    bool operator==(const BettingHistoryKey& other) const;
  };

  struct BettingHistoryKeyHash {
    size_t operator()(const BettingHistoryKey& key) const;
  };

  struct PublicStateKey {
    uint32_t betting_history_id = 0;
    PublicBucketId public_bucket = 0;

    bool operator==(const PublicStateKey& other) const;
  };

  struct PublicStateKeyHash {
    size_t operator()(const PublicStateKey& key) const;
  };

  struct ChanceTransitionKey {
    uint32_t parent_public_state_id = 0;
    PublicBucketId outcome_id = 0;

    bool operator==(const ChanceTransitionKey& other) const;
  };

  struct ChanceTransitionKeyHash {
    size_t operator()(const ChanceTransitionKey& key) const;
  };

  struct BettingHistoryRow {
    BettingHistoryRow() {
      action_ids.fill(0);
      action_child_ids.fill(kInvalidBettingHistoryId);
    }

    int street = 0;
    int pot = 0;
    std::array<int, 2> stack = {0, 0};
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    std::array<int, 2> player_contributions = {0, 0};
    int history_size = 0;
    // Only entries [0, history_size) are read when extending abstract history.
    std::array<int, BettingHistoryKey::kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;
    uint8_t action_count = 0;
    std::array<int, kMaxActionsPerNode> action_ids;
    std::array<uint32_t, kMaxActionsPerNode> action_child_ids;
    uint32_t chance_child_id = kInvalidBettingHistoryId;
  };

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct ChanceChildEntry {
    PublicBucketId outcome_id = 0;
    uint32_t public_state_id = kInvalidPublicStateId;
  };

  struct InfoSetAddress {
    uint32_t public_state_id = 0;
    int player = 0;
    PrivateBucketId private_bucket = 0;
  };

  struct PublicStateRow {
    PublicStateRow() {
      action_ids.fill(0);
      action_child_ids.fill(kInvalidPublicStateId);
    }

    BettingState betting;
    uint32_t betting_history_id = kInvalidBettingHistoryId;
    PublicBucketId public_bucket = 0;
    bool is_terminal = false;
    bool is_chance_node = false;
    int player_to_act = -1;
    uint8_t action_count = 0;
    std::array<GameAction, kMaxActionsPerNode> actions = {};
    std::array<int, kMaxActionsPerNode> action_ids = {};
    std::array<uint32_t, kMaxActionsPerNode> action_child_ids = {};
    uint32_t chance_child_offset = 0;
    uint32_t chance_child_count = 0;
  };

  static constexpr int kPrivateBucketChunkSize = 64;
  static constexpr int kPrivateBucketChunkCount =
      (kComboCount + kPrivateBucketChunkSize - 1) / kPrivateBucketChunkSize;

  struct PrivateRowChunk {
    PrivateRowChunk() { rows.fill(-1); }

    std::array<int32_t, kPrivateBucketChunkSize> rows;
  };

  struct PublicInfoSetSlabPlayer {
    std::array<std::unique_ptr<PrivateRowChunk>, kPrivateBucketChunkCount>
        private_row_chunks;
    std::vector<InfoSetRow> rows;
  };

  struct PublicInfoSetSlab {
    std::array<PublicInfoSetSlabPlayer, kPlayerCount> players;
  };

  using FrozenInfoSetActionOffsetRow =
      std::array<std::array<uint32_t, kComboCount>, kPlayerCount>;

  absl::flat_hash_map<BettingHistoryKey, uint32_t, BettingHistoryKeyHash>
      betting_history_ids;
  absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>
      public_state_ids;
  std::vector<PublicStateRow> public_state_rows;
  absl::flat_hash_map<ChanceTransitionKey, uint32_t, ChanceTransitionKeyHash>
      public_chance_child_ids;
  std::vector<ChanceChildEntry> chance_child_entries;
  std::vector<BettingHistoryRow> betting_history_rows;
  std::vector<FrozenInfoSetActionOffsetRow> frozen_info_set_action_offsets;
  size_t info_set_count = 0;
  std::vector<int> action_ids;
  std::vector<std::unique_ptr<PublicInfoSetSlab>> public_info_set_slabs;
};

struct MutableCumulativeArrays {
  std::vector<float, CacheLineAlignedAllocator<float>> cumulative_regrets;
  std::vector<float, CacheLineAlignedAllocator<float>> cumulative_strategies;
};

}  // namespace poker
