#include "src/strategy_tables.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>

namespace poker {
namespace {

void HashCombine(size_t& seed, uint64_t value) {
  seed ^= std::hash<uint64_t>{}(value) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2);
}

}  // namespace

bool StrategyTables::NodeKey::operator==(
    const NodeKey& other) const {
  return betting_node_id == other.betting_node_id &&
         board_bucket == other.board_bucket;
}

size_t StrategyTables::NodeKeyHash::operator()(
    const NodeKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.betting_node_id));
  HashCombine(seed, key.board_bucket);
  return seed;
}

bool StrategyTables::ChanceTransitionKey::operator==(
    const ChanceTransitionKey& other) const {
  return parent_node_id == other.parent_node_id &&
         outcome_id == other.outcome_id;
}

size_t StrategyTables::ChanceTransitionKeyHash::operator()(
    const ChanceTransitionKey& key) const {
  size_t seed = 0;
  HashCombine(seed, static_cast<uint64_t>(key.parent_node_id));
  HashCombine(seed, key.outcome_id);
  return seed;
}

std::optional<NodeId> StrategyTables::chance_child(
    NodeId parent_node_id,
    BoardBucketId outcome_id) const {
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
      first, last, outcome_id,
      [](const ChanceChildEntry& entry, BoardBucketId target) {
        return entry.outcome_id < target;
      });
  if (iter == last || iter->outcome_id != outcome_id) {
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
