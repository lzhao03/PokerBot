#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "src/betting_abstraction.h"
#include "src/card_abstraction.h"
#include "src/game_tree.h"
#include "src/poker_types.h"
#include "src/strategy_store.h"
#include "src/strategy_tables.h"

namespace poker {

struct TrainingRunStats {
  bool public_state_prebuild_complete = false;
  bool betting_history_transition_prebuild_complete = false;
  bool action_transition_prebuild_complete = false;
  bool chance_transition_prebuild_complete = false;
  bool info_set_prebuild_complete = false;
  bool private_bucket_prebuild_complete = false;
  bool frozen_info_set_lookup_prebuild_complete = false;
  int64_t prebuild_public_states = 0;
  int64_t prebuild_betting_histories = 0;
  int64_t prebuild_betting_history_transitions = 0;
  int64_t missing_betting_history_transitions = 0;
  int64_t prebuild_action_transitions = 0;
  int64_t missing_action_transitions = 0;
  int64_t prebuild_chance_transitions = 0;
  int64_t missing_chance_transitions = 0;
  int64_t prebuild_info_sets = 0;
  int64_t prebuild_action_entries = 0;
  int64_t prebuild_private_bucket_rows = 0;
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

class PublicStateGraph {
 public:
  using PublicBucketId = StrategyTables::PublicBucketId;
  using BettingHistoryKey = StrategyTables::BettingHistoryKey;
  using BettingHistoryRow = StrategyTables::BettingHistoryRow;
  using PublicStateKey = StrategyTables::PublicStateKey;
  using ChanceTransitionKey = StrategyTables::ChanceTransitionKey;
  using PublicStateRow = StrategyTables::PublicStateRow;

  static constexpr uint32_t kCappedPublicStateId =
      GameTree::kInvalidPublicStateId - 1;

  PublicStateGraph(const SolverConfig& config,
                   SolverStorage& storage,
                   GameTree& game_tree,
                   const CardAbstraction& card_abstraction,
                   const BettingAbstraction& betting_abstraction,
                   TraversalStats* stats);

  std::optional<uint32_t> get_or_create_public_state_row(
      const CompactPublicState& state);
  std::optional<uint32_t> action_child_public_state(
      uint32_t public_state_id,
      int action_index) const;
  std::optional<uint32_t> chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state) const;
  std::optional<uint32_t> get_or_create_action_child_public_state(
      uint32_t public_state_id,
      int action_index);
  std::optional<uint32_t> get_or_create_chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state);
  bool prebuild_public_state_rows(uint32_t root_public_state_id,
                                  int max_depth);
  bool validate_prebuilt_transitions(uint32_t root_public_state_id,
                                     int max_depth,
                                     TrainingRunStats& stats) const;

 private:
  const StrategyTables& tables() const { return storage_.frozen_ref(); }
  StrategyTables& mtables() { return storage_.mutable_ref(); }
  const std::vector<PublicStateRow>& rows() const {
    return tables().public_state_rows;
  }

  BettingHistoryKey make_betting_history_key(
      const CompactPublicState& state) const;
  BettingHistoryRow make_betting_history_row(
      const CompactPublicState& state) const;
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const CompactPublicState& state) const;
  PublicStateRow make_public_state_row(uint32_t betting_history_id,
                                       CompactPublicState state);
  uint32_t get_or_create_betting_history_id(const CompactPublicState& state);
  uint32_t get_or_create_betting_history_id(BettingHistoryKey key,
                                            BettingHistoryRow row);
  uint32_t get_or_create_action_child_betting_history_id(
      uint32_t parent_betting_history_id,
      int action_index,
      const CompactPublicState& child_state);
  uint32_t get_or_create_chance_child_betting_history_id(
      uint32_t parent_betting_history_id,
      const CompactPublicState& child_state);
  void cache_betting_history_actions(uint32_t betting_history_id,
                                     const PublicStateRow& row);
  std::optional<uint32_t> get_or_create_public_state_row(
      uint32_t betting_history_id,
      CompactPublicState state);
  template <typename Callback>
  bool for_each_required_chance_transition(const PublicStateRow& row,
                                           Callback&& callback) const;
  PublicBucketId chance_outcome_id(
      const CompactPublicState& child_state) const;
  void rebuild_chance_child_entries();

  const SolverConfig& config_;
  SolverStorage& storage_;
  GameTree& game_tree_;
  const CardAbstraction& card_abstraction_;
  const BettingAbstraction& betting_abstraction_;
  TraversalStats* stats_;
};

}  // namespace poker
