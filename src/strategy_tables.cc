#include "src/strategy_tables.h"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace poker {

std::optional<NodeId> StrategyTables::chance_child(
    NodeId parent_node_id,
    PublicObservationId child_public_observation) const {
  if (parent_node_id >= nodes.size()) {
    return std::nullopt;
  }
  const Node& graph_node = nodes[parent_node_id];
  const size_t begin = graph_node.chance_child_offset;
  const size_t end = begin + graph_node.chance_child_count;
  if (begin > chance_child_entries.size() || end > chance_child_entries.size()) {
    return std::nullopt;
  }

  const auto first =
      chance_child_entries.begin() + static_cast<std::ptrdiff_t>(begin);
  const auto last =
      chance_child_entries.begin() + static_cast<std::ptrdiff_t>(end);
  const auto iter = std::lower_bound(
      first, last, child_public_observation,
      [](const ChanceChildEntry& entry, PublicObservationId target) {
        return entry.child_public_observation < target;
      });
  if (iter == last ||
      iter->child_public_observation != child_public_observation) {
    return std::nullopt;
  }
  return iter->node_id;
}

std::optional<NodeId> StrategyTables::action_child(
    NodeId parent_node_id,
    int action_index) const {
  if (parent_node_id >= nodes.size()) {
    return std::nullopt;
  }
  const Node& graph_node = nodes[parent_node_id];
  if (graph_node.betting_node_id >= betting_nodes.size()) {
    return std::nullopt;
  }
  const BettingNode& node = betting_nodes[graph_node.betting_node_id];
  if (action_index < 0 || action_index >= node.action_count) {
    return std::nullopt;
  }

  const size_t child_index = static_cast<size_t>(graph_node.action_child_offset) +
                             static_cast<size_t>(action_index);
  if (child_index >= action_child_ids.size()) {
    return std::nullopt;
  }
  return action_child_ids[child_index];
}

}  // namespace poker
