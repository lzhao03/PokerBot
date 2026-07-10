#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/combo.h"
#include "src/poker_types.h"

#ifndef POKER_COARSE_PUBLIC_BUCKETS
#define POKER_COARSE_PUBLIC_BUCKETS 0
#endif

namespace poker {

inline constexpr uint32_t kInvalidPublicStateId =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kInvalidBettingNodeId =
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
  static constexpr uint32_t kPrivateBucketCount =
      POKER_COARSE_PUBLIC_BUCKETS != 0 ? 36 : kComboCount;
  static constexpr uint32_t kInvalidActionOffset =
      std::numeric_limits<uint32_t>::max();

  using BettingNodeId = uint32_t;

  enum class NodeKind : uint8_t {
    kDecision,
    kChance,
    kTerminal,
    kFrontier,
  };

  struct BettingEdge {
    GameAction action;
    BettingNodeId child = kInvalidBettingNodeId;
  };

  struct BettingNode {
    BettingState state;
    uint32_t action_begin = 0;
    uint8_t action_count = 0;
    BettingNodeId chance_child = kInvalidBettingNodeId;
    NodeKind kind = NodeKind::kDecision;
    int player_to_act = -1;
  };

  struct PublicStateKey {
    BettingNodeId betting_node_id = 0;
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
    PrivateBucketId private_bucket = 0;
  };

  struct PublicStateRow {
    BettingNodeId betting_node_id = kInvalidBettingNodeId;
    PublicBucketId public_bucket = 0;
    uint32_t action_child_offset = 0;
    uint32_t chance_child_offset = 0;
    uint32_t chance_child_count = 0;
  };

  std::optional<uint32_t> action_child(uint32_t parent_public_state_id,
                                       int action_index) const;
  std::optional<uint32_t> chance_child(
      uint32_t parent_public_state_id,
      PublicBucketId outcome_id) const;

  static constexpr int kPrivateBucketChunkSize = 64;
  static constexpr int kPrivateBucketChunkCount =
      (kPrivateBucketCount + kPrivateBucketChunkSize - 1) /
      kPrivateBucketChunkSize;

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
      std::array<uint32_t, kPrivateBucketCount>;

  BettingNodeId root_betting_node_id = kInvalidBettingNodeId;
  uint32_t root_public_state_id = kInvalidPublicStateId;
  std::vector<BettingNode> betting_nodes;
  std::vector<BettingEdge> betting_edges;
  absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>
      public_state_ids;
  std::vector<PublicStateRow> public_state_rows;
  std::vector<uint32_t> action_child_ids;
  absl::flat_hash_map<ChanceTransitionKey, uint32_t, ChanceTransitionKeyHash>
      public_chance_child_ids;
  std::vector<ChanceChildEntry> chance_child_entries;
  std::vector<FrozenInfoSetActionOffsetRow> frozen_info_set_action_offsets;
  size_t info_set_count = 0;
  std::vector<std::unique_ptr<PublicInfoSetSlab>> public_info_set_slabs;
};

struct MutableCumulativeArrays {
  std::vector<float, CacheLineAlignedAllocator<float>> cumulative_regrets;
  std::vector<float, CacheLineAlignedAllocator<float>> cumulative_strategies;
};

}  // namespace poker
