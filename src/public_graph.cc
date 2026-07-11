#include "src/public_graph.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace poker {

NodeId PublicGraph::action_child(NodeId parent, int action) const noexcept {
  if (parent >= nodes.size()) {
    return kInvalidNodeId;
  }
  const PublicNode& public_node = nodes[parent];
  if (public_node.betting_node_id >= betting_nodes.size()) {
    return kInvalidNodeId;
  }
  const BettingNode& betting_node =
      betting_nodes[public_node.betting_node_id];
  if (action < 0 || action >= betting_node.action_count) {
    return kInvalidNodeId;
  }
  const size_t child_index =
      static_cast<size_t>(public_node.action_child_begin) +
      static_cast<size_t>(action);
  if (child_index >= action_children.size()) {
    return kInvalidNodeId;
  }
  return action_children[child_index];
}

NodeId PublicGraph::required_chance_child(
    NodeId parent,
    PublicObservationId observation) const {
  if (parent >= nodes.size()) {
    throw std::logic_error("chance parent node is invalid");
  }
  const PublicNode& public_node = nodes[parent];
  const size_t begin = public_node.chance_child_begin;
  const size_t end = begin + public_node.chance_child_count;
  if (begin > chance_children.size() || end > chance_children.size()) {
    throw std::logic_error("chance child range is invalid");
  }

  const auto first =
      chance_children.begin() + static_cast<std::ptrdiff_t>(begin);
  const auto last =
      chance_children.begin() + static_cast<std::ptrdiff_t>(end);
  const auto child = std::lower_bound(
      first, last, observation,
      [](const ChanceChild& candidate, PublicObservationId target) {
        return candidate.observation < target;
      });
  if (child == last || child->observation != observation) {
    throw std::logic_error("required chance child is missing");
  }
  return child->child;
}

}  // namespace poker
