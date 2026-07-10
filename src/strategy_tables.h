#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/build_flags.h"
#include "src/combo.h"
#include "src/game_state.h"
#include "src/poker_types.h"

namespace poker {

using NodeId = uint32_t;

inline constexpr NodeId kInvalidNodeId =
    std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t kInvalidBettingNodeId =
    std::numeric_limits<uint32_t>::max();
inline constexpr NodeId kCappedNodeId = kInvalidNodeId - 1;

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
  using BoardBucketId = poker::BoardBucketId;
  static constexpr uint32_t kPrivateBucketCount =
      kCoarsePrivateBuckets ? 36 : kComboCount;
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
  };

  struct NodeKey {
    BettingNodeId betting_node_id = 0;
    BoardBucketId board_bucket = 0;

    friend bool operator==(const NodeKey&, const NodeKey&) = default;

    template <typename H>
    friend H AbslHashValue(H h, const NodeKey& key) {
      return H::combine(std::move(h), key.betting_node_id,
                        key.board_bucket);
    }
  };

  struct ChanceTransitionKey {
    NodeId parent_node_id = 0;
    BoardBucketId outcome_id = 0;

    friend bool operator==(const ChanceTransitionKey&,
                           const ChanceTransitionKey&) = default;

    template <typename H>
    friend H AbslHashValue(H h, const ChanceTransitionKey& key) {
      return H::combine(std::move(h), key.parent_node_id, key.outcome_id);
    }
  };

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct ChanceChildEntry {
    BoardBucketId outcome_id = 0;
    NodeId node_id = kInvalidNodeId;
  };

  struct InfoSetAddress {
    NodeId node_id = 0;
    PrivateBucketId private_bucket = 0;
  };

  struct Node {
    BettingNodeId betting_node_id = kInvalidBettingNodeId;
    BoardBucketId board_bucket = 0;
    uint32_t action_child_offset = 0;
    uint32_t chance_child_offset = 0;
    uint32_t chance_child_count = 0;
  };

  std::optional<NodeId> action_child(NodeId parent_node_id,
                                     int action_index) const;
  std::optional<NodeId> chance_child(
      NodeId parent_node_id,
      BoardBucketId outcome_id) const;

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
  NodeId root_node_id = kInvalidNodeId;
  std::vector<BettingNode> betting_nodes;
  std::vector<BettingEdge> betting_edges;
  absl::flat_hash_map<NodeKey, NodeId> node_ids;
  std::vector<Node> nodes;
  std::vector<NodeId> action_child_ids;
  absl::flat_hash_map<ChanceTransitionKey, NodeId> public_chance_child_ids;
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
