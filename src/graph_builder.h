#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "src/betting_abstraction.h"
#include "src/card_abstraction.h"
#include "src/poker_types.h"
#include "src/strategy_store.h"
#include "src/strategy_tables.h"

namespace poker {

struct TrainingRunStats {
  bool public_state_prebuild_complete = false;
  bool action_transition_prebuild_complete = false;
  bool chance_transition_prebuild_complete = false;
  bool info_set_prebuild_complete = false;
  bool frozen_info_set_lookup_prebuild_complete = false;
  int64_t prebuild_public_states = 0;
  int64_t prebuild_action_transitions = 0;
  int64_t missing_action_transitions = 0;
  int64_t prebuild_chance_transitions = 0;
  int64_t missing_chance_transitions = 0;
  int64_t prebuild_info_sets = 0;
  int64_t prebuild_action_entries = 0;
  int64_t prebuild_frozen_info_set_lookup_rows = 0;
  double prebuild_seconds = 0.0;
  double info_set_prebuild_seconds = 0.0;
  int warmup_iterations = 0;
  int frozen_iterations = 0;
  double warmup_seconds = 0.0;
  double frozen_seconds = 0.0;
  int64_t warmup_cfr_updates = 0;
  int64_t frozen_cfr_updates = 0;
};

class GraphBuilder {
 public:
  using Node = StrategyTables::Node;

  GraphBuilder(const SolverConfig& config,
               SolverStorage& storage,
               const BettingAbstraction& betting_abstraction,
               TraversalStats& stats);

  std::optional<NodeId> get_or_create_node(const ExactGameState& state);
  std::optional<NodeId> get_or_create_action_child(
      NodeId parent_node_id,
      int action_index,
      const Board& parent_board);
  std::optional<NodeId> get_or_create_chance_child(
      NodeId parent_node_id,
      const ExactGameState& child_state);
  bool prebuild_reachable_nodes(NodeId root_id,
                                const Board& root_board,
                                int max_depth,
                                std::vector<std::optional<Board>>& node_boards);
  bool validate_prebuilt_nodes(NodeId root_id,
                               const Board& root_board,
                               int max_depth,
                               TrainingRunStats& stats) const;

 private:
  using BoardBucketId = StrategyTables::BoardBucketId;
  using BettingNodeId = StrategyTables::BettingNodeId;
  using BettingNode = StrategyTables::BettingNode;
  using BettingEdge = StrategyTables::BettingEdge;
  using NodeKey = StrategyTables::NodeKey;
  using ChanceTransitionKey = StrategyTables::ChanceTransitionKey;

  const StrategyTables& tables() const { return storage_.frozen_ref(); }
  StrategyTables& mtables() { return storage_.mutable_ref(); }
  const std::vector<Node>& nodes() const {
    return tables().nodes;
  }

  BettingNodeId append_betting_node(const BettingState& state);
  BettingNodeId get_or_create_root_betting_node(const BettingState& state);
  BettingNodeId get_or_create_action_betting_child(
      BettingNodeId parent_node_id,
      int action_index);
  BettingNodeId get_or_create_chance_betting_child(
      BettingNodeId parent_node_id,
      const BettingState& child_state);
  NodeKey node_key(BettingNodeId betting_node_id,
                   StreetKind street,
                   const Board& board) const;
  std::optional<NodeId> find_node(
      BettingNodeId betting_node_id,
      StreetKind street,
      const Board& board) const;
  Node make_node(BettingNodeId betting_node_id,
                 const ExactGameState& state);
  std::optional<NodeId> get_or_create_node(
      BettingNodeId betting_node_id,
      const ExactGameState& state);
  std::optional<NodeId> find_or_cache_action_child(
      NodeId parent_node_id,
      int action_index);
  std::optional<NodeId> find_or_cache_chance_child(
      NodeId parent_node_id,
      const ExactGameState& child_state);
  bool node_limit_reached() const;
  bool can_insert_node() const;
  template <typename Callback>
  bool for_each_required_chance_transition(const Node& node,
                                           const Board& board,
                                           Callback&& callback) const;
  BoardBucketId chance_outcome_id(
      const ExactGameState& child_state) const;
  void rebuild_chance_child_entries();

  const SolverConfig& config_;
  SolverStorage& storage_;
  const BettingAbstraction& betting_abstraction_;
  TraversalStats& stats_;
};

}  // namespace poker
