#pragma once

#include <cstdint>
#include <limits>
#include <vector>

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

class PublicGraph {
 public:
  using NodeId = poker::NodeId;
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

  struct PublicNode {
    BettingNodeId betting_node_id = kInvalidBettingNodeId;
    PublicObservationId public_observation = 0;
    uint32_t action_child_begin = 0;
    uint32_t chance_child_begin = 0;
    uint32_t chance_child_count = 0;
  };

  struct ChanceChild {
    PublicObservationId observation = 0;
    NodeId child = kInvalidNodeId;
  };

  NodeId action_child(NodeId parent, int action) const noexcept;
  NodeId required_chance_child(
      NodeId parent,
      PublicObservationId observation) const;

  BettingNodeId root_betting_node = kInvalidBettingNodeId;
  NodeId root = kInvalidNodeId;
  std::vector<BettingNode> betting_nodes;
  std::vector<BettingEdge> betting_edges;
  std::vector<PublicNode> nodes;
  std::vector<NodeId> action_children;
  std::vector<ChanceChild> chance_children;
};

}  // namespace poker
