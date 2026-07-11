#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "src/game_state.h"
#include "src/poker_types.h"

namespace poker {

using NodeId = uint32_t;
using BettingNodeId = uint32_t;

inline constexpr NodeId kInvalidNodeId =
    std::numeric_limits<uint32_t>::max();
inline constexpr BettingNodeId kInvalidBettingNodeId =
    std::numeric_limits<uint32_t>::max();
inline constexpr NodeId kCappedNodeId = kInvalidNodeId - 1;

class StrategyTables {
 public:
  static constexpr uint32_t kInvalidActionOffset =
      std::numeric_limits<uint32_t>::max();

  using BettingNodeId = poker::BettingNodeId;

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
    BettingNodeId betting_history_id = 0;
    PublicObservationId public_observation = 0;

    friend bool operator==(const NodeKey&, const NodeKey&) = default;

    template <typename H>
    friend H AbslHashValue(H h, const NodeKey& key) {
      return H::combine(std::move(h), key.betting_history_id,
                        key.public_observation);
    }
  };

  struct ChanceTransitionKey {
    NodeId parent_node_id = 0;
    PublicObservationId child_public_observation = 0;

    friend bool operator==(const ChanceTransitionKey&,
                           const ChanceTransitionKey&) = default;

    template <typename H>
    friend H AbslHashValue(H h, const ChanceTransitionKey& key) {
      return H::combine(std::move(h), key.parent_node_id,
                        key.child_public_observation);
    }
  };

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct ChanceChildEntry {
    PublicObservationId child_public_observation = 0;
    NodeId node_id = kInvalidNodeId;
  };

  struct InfoSetKey {
    NodeId node_id = 0;
    PrivateObservationId private_observation = 0;
  };

  struct GrowingPublicInfoSets {
    absl::flat_hash_map<PrivateObservationId, InfoSetRow> rows;
  };

  struct FrozenInfoSetEntry {
    PrivateObservationId private_observation = 0;
    uint32_t action_offset = kInvalidActionOffset;
  };

  struct FrozenPublicInfoSetRange {
    uint32_t begin = 0;
    uint32_t count = 0;
  };

  struct Node {
    BettingNodeId betting_node_id = kInvalidBettingNodeId;
    PublicObservationId public_observation = 0;
    uint32_t action_child_offset = 0;
    uint32_t chance_child_offset = 0;
    uint32_t chance_child_count = 0;
  };

  std::optional<NodeId> action_child(NodeId parent_node_id,
                                     int action_index) const;
  std::optional<NodeId> chance_child(
      NodeId parent_node_id,
      PublicObservationId child_public_observation) const;

  BettingNodeId root_betting_node_id = kInvalidBettingNodeId;
  NodeId root_node_id = kInvalidNodeId;
  std::vector<BettingNode> betting_nodes;
  std::vector<BettingEdge> betting_edges;
  absl::flat_hash_map<NodeKey, NodeId> node_ids;
  std::vector<Node> nodes;
  std::vector<NodeId> action_child_ids;
  absl::flat_hash_map<ChanceTransitionKey, NodeId> public_chance_child_ids;
  std::vector<ChanceChildEntry> chance_child_entries;
  std::vector<FrozenInfoSetEntry> frozen_info_set_entries;
  std::vector<FrozenPublicInfoSetRange> frozen_info_set_ranges;
  size_t info_set_count = 0;
  std::vector<std::unique_ptr<GrowingPublicInfoSets>> growing_info_sets;
};

}  // namespace poker
