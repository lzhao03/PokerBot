#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "src/betting_abstraction.h"
#include "src/card_abstraction.h"
#include "src/game_state.h"
#include "src/poker_types.h"
#include "src/solver_stats.h"
#include "src/strategy_store.h"
#include "src/strategy_tables.h"

namespace poker {

struct GraphBuilderTestAccess;

struct PublicNodeKey {
  BettingNodeId betting_history_id = 0;
  PublicObservationId public_observation = 0;

  friend bool operator==(const PublicNodeKey&, const PublicNodeKey&) = default;

  template <typename H>
  friend H AbslHashValue(H h, const PublicNodeKey& key) {
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

struct PublicGraphBuildState {
  absl::flat_hash_map<PublicNodeKey, NodeId> node_ids;
  absl::flat_hash_map<ChanceTransitionKey, NodeId> chance_children;
};

class GraphBuilder {
 public:
  using Node = PublicGraph::PublicNode;

  GraphBuilder(const SolverConfig& config,
               const BettingRules& rules,
               SolverStorage& storage,
               const BettingAbstraction& betting_abstraction,
               TraversalStats& stats);

  std::optional<NodeId> get_or_create_node(const ExactPublicState& state);
  std::optional<NodeId> get_or_create_action_child(
      NodeId parent_node_id,
      int action_index,
      const BoardRunout& parent_board);
  std::optional<NodeId> get_or_create_chance_child(
      NodeId parent_node_id,
      const ExactPublicState& child_state);
  bool prebuild_reachable_nodes(NodeId root_id,
                                const BoardRunout& root_board,
                                int max_depth,
                                std::vector<std::optional<BoardRunout>>& node_boards);
  bool validate_prebuilt_nodes(NodeId root_id,
                               const BoardRunout& root_board,
                               int max_depth,
                               TrainingRunStats& stats) const;

 private:
  friend struct GraphBuilderTestAccess;

  using BettingNodeId = poker::BettingNodeId;
  using BettingNode = PublicGraph::BettingNode;
  using BettingEdge = PublicGraph::BettingEdge;

  const StrategyTables& tables() const { return storage_.frozen_ref(); }
  StrategyTables& mtables() { return storage_.mutable_ref(); }
  const PublicGraph& graph() const { return tables().graph; }
  PublicGraph& mgraph() { return mtables().graph; }
  const std::vector<Node>& nodes() const {
    return graph().nodes;
  }

  BettingNodeId append_betting_node(const BettingState& state);
  BettingNodeId get_or_create_root_betting_node(const BettingState& state);
  BettingNodeId get_or_create_action_betting_child(
      BettingNodeId parent_node_id,
      int action_index);
  BettingNodeId get_or_create_chance_betting_child(
      BettingNodeId parent_node_id,
      const BettingState& child_state);
  PublicNodeKey node_key(BettingNodeId betting_node_id,
                         PublicObservationId public_observation) const;
  std::optional<NodeId> find_node(
      BettingNodeId betting_node_id,
      PublicObservationId public_observation) const;
  Node make_node(BettingNodeId betting_node_id,
                 const ExactPublicState& state,
                 PublicObservationId public_observation);
  std::optional<NodeId> get_or_create_node(
      BettingNodeId betting_node_id,
      const ExactPublicState& state,
      PublicObservationId public_observation);
  std::optional<NodeId> find_or_cache_action_child(
      NodeId parent_node_id,
      int action_index);
  std::optional<NodeId> find_or_cache_chance_child(
      NodeId parent_node_id,
      const ExactPublicState& child_state);
  bool node_limit_reached() const;
  bool can_insert_node() const;
  template <typename Callback>
  bool for_each_required_chance_transition(const Node& node,
                                           const BoardRunout& board,
                                           Callback&& callback) const;
  void rebuild_chance_child_entries();

  const SolverConfig& config_;
  const BettingRules& rules_;
  SolverStorage& storage_;
  const BettingAbstraction& betting_abstraction_;
  TraversalStats& stats_;
  PublicGraphBuildState build_state_;
};

}  // namespace poker
