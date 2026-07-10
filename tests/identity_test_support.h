#pragma once

#include <stdexcept>
#include <vector>

#include "absl/types/span.h"
#include "src/card_abstraction.h"
#include "src/graph_builder.h"

namespace poker {

struct GraphBuilderTestAccess {
  explicit GraphBuilderTestAccess(GraphBuilder& graph) : graph_(graph) {}

  NodeId root_node() const {
    return graph_.tables().root_node_id;
  }

  NodeId action_child(NodeId parent, int action) const {
    const auto child = graph_.tables().action_child(parent, action);
    if (!child.has_value()) {
      throw std::logic_error("missing action child");
    }
    return *child;
  }

  NodeId chance_child(NodeId parent,
                      PublicObservationId observation) const {
    const StrategyTables::ChanceTransitionKey key{parent, observation};
    const auto& children = graph_.tables().public_chance_child_ids;
    const auto child = children.find(key);
    if (child == children.end()) {
      throw std::logic_error("missing chance child");
    }
    return child->second;
  }

  NodeId frozen_chance_child(NodeId parent,
                             PublicObservationId observation) const {
    const auto child = graph_.tables().chance_child(parent, observation);
    if (!child.has_value()) {
      throw std::logic_error("missing frozen chance child");
    }
    return *child;
  }

  PublicObservationId public_observation(NodeId node) const {
    return graph_node(node).public_observation;
  }

  BettingNodeId betting_history(NodeId node) const {
    return graph_node(node).betting_node_id;
  }

  void freeze() {
    graph_.rebuild_chance_child_entries();
    graph_.storage_.freeze();
  }

  void set_public_observation(NodeId node,
                              PublicObservationId observation) {
    graph_.mtables().nodes.at(node).public_observation = observation;
  }

 private:
  const StrategyTables::Node& graph_node(NodeId node) const {
    const auto& nodes = graph_.tables().nodes;
    if (node >= nodes.size()) {
      throw std::logic_error("invalid graph node");
    }
    return nodes[node];
  }

  GraphBuilder& graph_;
};

namespace test {

using PublicObservationTrace = std::vector<PublicObservationId>;
using PrivateObservationTrace = std::vector<PrivateBucketId>;

inline SolverConfig IdentityConfig() {
  SolverConfig config;
  config.bet_sizes = {0.5, 1.0, 1.25};
  config.starting_stack_size = 20;
  config.small_blind = 1;
  config.big_blind = 2;
  return config;
}

class IdentityGraph {
 public:
  IdentityGraph()
      : config_(IdentityConfig()),
        betting_(config_),
        graph_(config_, rules_, storage_, betting_, stats_),
        access_(graph_) {}

  ExactPublicState initial_state() const {
    return MakeInitialState(rules_, {20, 20}, {1, 2});
  }

  NodeId root(const ExactPublicState& state) {
    const auto node = graph_.get_or_create_node(state);
    if (!node.has_value()) {
      throw std::logic_error("failed to create root node");
    }
    return *node;
  }

  NodeId action_child(NodeId parent,
                      ExactPublicState& state,
                      GameAction action) {
    const ActionMenu menu = betting_.actions_for_betting_node(state.betting);
    int action_index = -1;
    for (uint8_t i = 0; i < menu.count; ++i) {
      if (menu.actions[i] == action) {
        action_index = i;
        break;
      }
    }
    if (action_index < 0) {
      throw std::logic_error("action is absent from abstraction menu");
    }

    const auto child = graph_.get_or_create_action_child(
        parent, action_index, state.board);
    if (!child.has_value()) {
      throw std::logic_error("failed to create action child");
    }
    state.betting = ApplyAction(state.betting, action);
    if (access_.action_child(parent, action_index) != *child) {
      throw std::logic_error("action child accessor mismatch");
    }
    return *child;
  }

  NodeId chance_child(NodeId parent,
                      ExactPublicState& state,
                      absl::Span<const CardId> cards) {
    state = ApplyChance(state, cards, rules_);
    const auto child = graph_.get_or_create_chance_child(parent, state);
    if (!child.has_value()) {
      throw std::logic_error("failed to create chance child");
    }
    const PublicObservationId observation =
        public_observation_id(state.betting.street, state.board);
    if (access_.chance_child(parent, observation) != *child) {
      throw std::logic_error("chance child accessor mismatch");
    }
    return *child;
  }

  const GraphBuilderTestAccess& access() const { return access_; }
  GraphBuilderTestAccess& access() { return access_; }

  bool prebuild(NodeId root, const BoardRunout& board, int max_depth) {
    std::vector<std::optional<BoardRunout>> node_boards;
    return graph_.prebuild_reachable_nodes(root, board, max_depth,
                                           node_boards);
  }

  bool validate(NodeId root, const BoardRunout& board, int max_depth) const {
    TrainingRunStats stats;
    return graph_.validate_prebuilt_nodes(root, board, max_depth, stats);
  }

 private:
  SolverConfig config_;
  BettingRules rules_{2};
  SolverStorage storage_;
  BettingAbstraction betting_;
  TraversalStats stats_;
  GraphBuilder graph_;
  GraphBuilderTestAccess access_;
};

inline PublicObservationId PublicObservation(
    const ExactPublicState& state) {
  return public_observation_id(state.betting.street, state.board);
}

inline PrivateBucketId PrivateObservation(
    ComboId hand,
    const ExactPublicState& state) {
  return private_bucket(hand, state.betting.street,
                        board_features(state.board));
}

inline NodeId ClosePreflop(IdentityGraph& graph,
                           ExactPublicState& state) {
  NodeId node = graph.root(state);
  node = graph.action_child(node, state, {ActionKind::kCall, 2});
  return graph.action_child(node, state, {ActionKind::kCheck, 0});
}

inline NodeId CheckCheck(IdentityGraph& graph,
                         NodeId node,
                         ExactPublicState& state) {
  node = graph.action_child(node, state, {ActionKind::kCheck, 0});
  return graph.action_child(node, state, {ActionKind::kCheck, 0});
}

}  // namespace test
}  // namespace poker
