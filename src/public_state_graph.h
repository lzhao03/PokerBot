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
  using PublicStateRow = StrategyTables::PublicStateRow;

  GraphBuilder(const SolverConfig& config,
               SolverStorage& storage,
               const BettingAbstraction& betting_abstraction,
               TraversalStats& stats);

  std::optional<uint32_t> get_or_create_row(const ExactGameState& state);
  std::optional<uint32_t> get_or_create_action_child(
      uint32_t parent_public_state_id,
      int action_index,
      const Board& parent_board);
  std::optional<uint32_t> get_or_create_chance_child(
      uint32_t parent_public_state_id,
      const ExactGameState& child_state);
  bool prebuild_reachable_rows(uint32_t root_id,
                               const Board& root_board,
                               int max_depth,
                               std::vector<std::optional<Board>>& row_boards);
  bool validate_prebuilt_rows(uint32_t root_id,
                              const Board& root_board,
                              int max_depth,
                              TrainingRunStats& stats) const;

 private:
  using PublicBucketId = StrategyTables::PublicBucketId;
  using BettingNodeId = StrategyTables::BettingNodeId;
  using BettingNode = StrategyTables::BettingNode;
  using BettingEdge = StrategyTables::BettingEdge;
  using PublicStateKey = StrategyTables::PublicStateKey;
  using ChanceTransitionKey = StrategyTables::ChanceTransitionKey;

  const StrategyTables& tables() const { return storage_.frozen_ref(); }
  StrategyTables& mtables() { return storage_.mutable_ref(); }
  const std::vector<PublicStateRow>& rows() const {
    return tables().public_state_rows;
  }

  BettingNodeId append_betting_node(const BettingState& state);
  BettingNodeId get_or_create_root_betting_node(const BettingState& state);
  BettingNodeId get_or_create_action_betting_child(
      BettingNodeId parent_node_id,
      int action_index);
  BettingNodeId get_or_create_chance_betting_child(
      BettingNodeId parent_node_id,
      const BettingState& child_state);
  PublicStateKey row_key(BettingNodeId betting_node_id,
                         StreetKind street,
                         const Board& board) const;
  std::optional<uint32_t> find_row(
      BettingNodeId betting_node_id,
      StreetKind street,
      const Board& board) const;
  PublicStateRow make_row(BettingNodeId betting_node_id,
                          const ExactGameState& state);
  std::optional<uint32_t> get_or_create_row(
      BettingNodeId betting_node_id,
      const ExactGameState& state);
  std::optional<uint32_t> find_or_cache_action_child(
      uint32_t parent_public_state_id,
      int action_index);
  std::optional<uint32_t> find_or_cache_chance_child(
      uint32_t parent_public_state_id,
      const ExactGameState& child_state);
  bool row_limit_reached() const;
  bool can_insert_row() const;
  template <typename Callback>
  bool for_each_required_chance_transition(const PublicStateRow& row,
                                           const Board& board,
                                           Callback&& callback) const;
  PublicBucketId chance_outcome_id(
      const ExactGameState& child_state) const;
  void rebuild_chance_child_entries();

  const SolverConfig& config_;
  SolverStorage& storage_;
  const BettingAbstraction& betting_abstraction_;
  TraversalStats& stats_;
};

}  // namespace poker
