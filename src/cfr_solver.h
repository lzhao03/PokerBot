#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "src/card_abstraction.h"
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"
#include "src/training_range.h"

namespace poker {

class TerminalUtilityCache;
class CFRSolver {
public:
  struct TraversalStats {
    int64_t cfr_updates = 0;
    int64_t preflop_updates = 0;
    int64_t flop_updates = 0;
    int64_t turn_updates = 0;
    int64_t river_updates = 0;
    int max_decision_depth = 0;
    int64_t child_nodes_created = 0;
    int64_t chance_samples = 0;
    int64_t terminal_utility_calls = 0;
    int64_t fold_utility_calls = 0;
    int64_t showdown_utility_calls = 0;
    int64_t action_entry_touches = 0;
    int64_t atomic_regret_update_retries = 0;
    int64_t betting_history_transition_hits = 0;
    int64_t betting_history_transition_misses = 0;
  };

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

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

  using PrivateBucketId = FrozenStrategyTables::PrivateBucketId;

  CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config, const GameState& initial_state);
  
  // Run CFR for a specified number of iterations
  void run(int iterations, ComboId player_a_hand, ComboId player_b_hand);
  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);
  
  double evaluate_strategy(ComboId player_a_hand, ComboId player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);
  
  // Get the expected value of the game for a player
  double get_expected_value(int player_id) const;
  int get_iterations_run() const { return iterations_run_; }
  int64_t get_cfr_update_count() const {
    return cfr_update_count_;
  }
  size_t get_info_set_count() const { return frozen_tables_->info_set_count; }
  size_t get_public_state_count() const {
    return frozen_tables_->public_state_rows.size();
  }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  TrainingRunStats get_last_training_run_stats() const {
    return last_training_run_stats_;
  }
  void add_traversal_stats(const TraversalStats& stats);
  UtilityCacheStats get_utility_cache_stats() const;
  static bool traversal_stats_enabled();
private:
  friend class CFRSolverRegretTestPeer;

  struct RangeDeal {
    RangeDeal(ComboId player_a_combo, ComboId player_b_combo)
        : player_a_combo(player_a_combo),
          player_b_combo(player_b_combo) {}

    ComboId player_a_combo = 0;
    ComboId player_b_combo = 0;
  };

  struct RangeSampler {
    RangeSampler(const TrainingRange& player_a_range,
                 const TrainingRange& player_b_range);

    RangeDeal sample(std::mt19937& rng) const;

    const TrainingRange& player_a_range;
    const TrainingRange& player_b_range;
    std::vector<float> compatible_player_b_weight;
    std::vector<float> player_a_sample_weights;
    std::vector<float> player_a_cumulative_weights;
    float total_player_a_weight = 0.0f;
    std::vector<uint32_t> compatible_player_b_offsets;
    std::vector<uint16_t> compatible_player_b_counts;
    std::vector<ComboId> compatible_player_b_combos;
    std::vector<float> compatible_player_b_cumulative_weights;
  };

  struct PrivateCards {
    static PrivateCards FromCombo(ComboId combo_id);

    CardMask mask() const;

    ComboId combo = 0;
  };

  struct SampledChanceTransition {
    uint32_t child_public_state_id = GameTree::Node::kInvalidPublicStateId;
    CompactPublicState exact_child_state;
  };

  using OptionalTrainingRange =
      std::optional<std::reference_wrapper<const TrainingRangeView>>;

  static constexpr uint32_t kCappedPublicStateId =
      GameTree::Node::kInvalidPublicStateId - 1;

  using ConditionedRanges = std::vector<TrainingRangeView>;
  using StrategyProbabilities = absl::InlinedVector<double, 8>;

  struct RangeScratchFrame {
    ConditionedRanges conditioned_ranges;
    TrainingRangeView public_player_a_range;
    TrainingRangeView public_player_b_range;
  };

  struct TraversalScratch {
    void reserve_depth(size_t depth_count) { frames.reserve(depth_count); }

    RangeScratchFrame& frame(size_t depth) {
      if (depth >= frames.capacity()) {
        throw std::logic_error("TraversalScratch depth reserve exhausted");
      }
      while (frames.size() <= depth) {
        frames.emplace_back();
      }
      return frames[depth];
    }

    std::vector<RangeScratchFrame> frames;
  };

  using BettingHistoryKey = FrozenStrategyTables::BettingHistoryKey;
  using BettingHistoryKeyHash = FrozenStrategyTables::BettingHistoryKeyHash;
  using PublicBucketId = FrozenStrategyTables::PublicBucketId;
  using PublicStateKey = FrozenStrategyTables::PublicStateKey;
  using PublicStateKeyHash = FrozenStrategyTables::PublicStateKeyHash;
  using BettingHistoryRow = FrozenStrategyTables::BettingHistoryRow;

  using InfoSetRow = FrozenStrategyTables::InfoSetRow;
  using ChanceChildEntry = FrozenStrategyTables::ChanceChildEntry;
  using InfoSetAddress = FrozenStrategyTables::InfoSetAddress;
  using PublicStateRow = FrozenStrategyTables::PublicStateRow;
  using PrivateRowChunk = FrozenStrategyTables::PrivateRowChunk;
  using PublicInfoSetSlabPlayer = FrozenStrategyTables::PublicInfoSetSlabPlayer;
  using PublicInfoSetSlab = FrozenStrategyTables::PublicInfoSetSlab;
  static constexpr int kPrivateBucketChunkSize =
      FrozenStrategyTables::kPrivateBucketChunkSize;

  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache);
  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache,
            GameState initial_state);

  SolverConfig config_;
  GameState initial_state_;
  std::shared_ptr<GameTree> game_tree_;
  std::mt19937 rng_;
  double cumulative_root_utility_;
  int iterations_run_ = 0;
  int64_t cfr_update_count_ = 0;
  TraversalStats traversal_stats_;
  TrainingRunStats last_training_run_stats_;
  std::shared_ptr<TerminalUtilityCache> utility_cache_;
  CardAbstraction card_abstraction_;
  // Set to true after warmup; workers may only write cumulative arrays.
  bool frozen_ = false;
  // True only when prebuild validation proved frozen child rows are complete.
  bool require_frozen_children_ = false;
  std::shared_ptr<FrozenStrategyTables> mutable_tables_;
  std::shared_ptr<const FrozenStrategyTables> frozen_tables_;
  std::shared_ptr<MutableCumulativeArrays> cumulative_;
  
  // Helper methods
  void run_iterations(int iterations,
                      const HandRange& player_a_range,
                      const HandRange& player_b_range);
  void run_frozen_iterations(int iterations,
                             int num_threads,
                             uint32_t root_public_state_id,
                             const RangeSampler& range_sampler,
                             const TrainingRange& player_a_training_range,
                             const TrainingRange& player_b_training_range);
  double cfr_with_ranges(
      uint32_t public_state_id,
      const CompactPublicState& state,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range);
  struct ExactBoardState {
    std::array<CardId, kMaxBoardCards> cards = {};
    uint8_t count = 0;
    CardMask mask = 0;
  };
  struct SampledFrozenChanceTransition {
    uint32_t child_public_state_id = GameTree::Node::kInvalidPublicStateId;
    ExactBoardState child_board;
  };
  static ExactBoardState exact_board_from_state(
      const CompactPublicState& state);
  static CompactPublicState state_with_exact_board(
      CompactPublicState state,
      const ExactBoardState& exact_board);
  double cfr_frozen_regret_only(
      uint32_t public_state_id,
      const ExactBoardState& exact_board,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int depth,
      bool use_atomic_updates);
  void average_strategy_probabilities(
      uint32_t public_state_id,
      const PublicStateRow& row,
      int player,
      const PrivateCards& private_cards,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const InfoSetRow& info_set_row,
      const PublicStateRow& public_state_row,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  void condition_ranges_for_actions(
      const TrainingRangeView& range,
      const CompactPublicState& state,
      uint32_t public_state_id,
      int player,
      const int* action_ids,
      size_t action_count,
      ConditionedRanges& conditioned_ranges);
  BettingHistoryKey make_betting_history_key(const GameState& state) const;
  BettingHistoryKey make_betting_history_key(
      const CompactPublicState& state) const;
  BettingHistoryRow make_betting_history_row(const GameState& state) const;
  BettingHistoryRow make_betting_history_row(
      const CompactPublicState& state) const;
  uint32_t get_or_create_betting_history_id(BettingHistoryKey key,
                                            BettingHistoryRow row);
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const GameState& state) const;
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const CompactPublicState& state) const;
  uint32_t get_or_create_betting_history_id(const GameState& state);
  uint32_t get_or_create_betting_history_id(
      const CompactPublicState& state);
  uint32_t get_or_create_action_child_betting_history_id(
      uint32_t parent_betting_history_id,
      int action_index,
      const CompactPublicState& child_state);
  uint32_t get_or_create_chance_child_betting_history_id(
      uint32_t parent_betting_history_id,
      const CompactPublicState& child_state);
  void cache_betting_history_actions(uint32_t betting_history_id,
                                     const PublicStateRow& row);
  CompactPublicState compact_public_state_from_game_state(
      const GameState& state);
  GameState materialize_game_state(const CompactPublicState& state) const;
  PublicStateRow make_public_state_row(uint32_t betting_history_id,
                                       const GameState& state);
  PublicStateRow make_public_state_row(uint32_t betting_history_id,
                                       CompactPublicState state);
  std::optional<uint32_t> get_or_create_public_state_row(
      uint32_t betting_history_id,
      const GameState& state);
  std::optional<uint32_t> get_or_create_public_state_row(
      uint32_t betting_history_id,
      CompactPublicState state);
  std::optional<uint32_t> get_or_create_public_state_row(
      const GameState& state);
  std::optional<uint32_t> action_child_public_state(
      uint32_t public_state_id,
      int action_index) const;
  uint32_t required_action_child_public_state(uint32_t public_state_id,
                                              int action_index) const;
  uint32_t strict_action_child_public_state(const PublicStateRow& row,
                                            size_t action_index) const;
  std::optional<uint32_t> chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state,
      absl::Span<const CardId> cards) const;
  uint32_t required_chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state,
      absl::Span<const CardId> cards) const;
  uint32_t strict_chance_child_public_state(
      const PublicStateRow& row,
      const CompactPublicState& child_state,
      absl::Span<const CardId> cards) const;
  std::optional<uint32_t> chance_child_public_state(
      uint32_t public_state_id,
      absl::Span<const CardId> cards) const;
  bool for_each_required_chance_transition(
      const PublicStateRow& row,
      const std::function<bool(const CompactPublicState&,
                               absl::Span<const CardId>)>& callback) const;
  int chance_child_lookup_key(const PublicStateRow& row,
                              const CompactPublicState& child_state,
                              absl::Span<const CardId> cards) const;
  std::optional<uint32_t> get_or_create_action_child_public_state(
      uint32_t public_state_id,
      int action_index);
  std::optional<uint32_t> get_or_create_chance_child_public_state(
      uint32_t public_state_id,
      const CompactPublicState& child_state,
      absl::Span<const CardId> cards);
  std::optional<uint32_t> get_or_create_chance_child_public_state(
      uint32_t public_state_id,
      absl::Span<const CardId> cards);
  bool prebuild_public_state_rows(uint32_t root_public_state_id,
                                  int max_depth);
  void rebuild_chance_child_entries();
  struct PrebuildValidationStats {
    bool betting_history_transition_prebuild_complete = false;
    bool action_transition_prebuild_complete = false;
    bool chance_transition_prebuild_complete = false;
    int64_t prebuild_betting_history_transitions = 0;
    int64_t missing_betting_history_transitions = 0;
    int64_t prebuild_action_transitions = 0;
    int64_t missing_action_transitions = 0;
    int64_t prebuild_chance_transitions = 0;
    int64_t missing_chance_transitions = 0;
  };

  PrebuildValidationStats validate_prebuilt_transitions(
      uint32_t root_public_state_id,
      int max_depth) const;
  bool prebuild_info_set_rows(const TrainingRangeView& player_a_range,
                              const TrainingRangeView& player_b_range);
  bool prebuild_private_bucket_rows();
  bool prebuild_frozen_info_set_action_offsets();
  PrivateBucketId private_bucket_for_frozen_row(uint32_t public_state_id,
                                                ComboId combo_id) const;
  uint32_t frozen_info_set_action_offset(uint32_t public_state_id,
                                         int player,
                                         PrivateBucketId private_bucket) const;
  const InfoSetRow* get_or_create_info_set_row(
      InfoSetAddress address,
      absl::Span<const int> action_ids);
  std::optional<uint32_t> strategy_betting_history_id(
      const GameState& state) const;
  FrozenStrategyTables& mutable_tables();
  InfoSetRow append_info_set_actions(absl::Span<const int> action_ids);
  PublicInfoSetSlab& get_or_create_public_info_set_slab(
      uint32_t public_state_id);
  const PublicInfoSetSlab* public_info_set_slab(
      uint32_t public_state_id) const;
  const InfoSetRow* find_info_set_row(InfoSetAddress address) const;
  static const InfoSetRow* find_info_set_row(
      const PublicInfoSetSlabPlayer& player_slab,
      PrivateBucketId private_bucket);
  static int32_t& get_or_create_private_row_slot(
      PublicInfoSetSlabPlayer& player_slab,
      PrivateBucketId private_bucket);
  double utility(const GameState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double utility(const CompactPublicState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double frozen_utility(const PublicStateRow& row,
                        const ExactBoardState& exact_board,
                        const PrivateCards& player_a_cards,
                        const PrivateCards& player_b_cards);
  double uncached_utility(const GameState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double uncached_utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(uint32_t public_state_id,
                                const CompactPublicState& state,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards);
  double evaluate_strategy_samples(
      int samples,
      uint32_t root_public_state_id,
      RangeSampler range_sampler);
  void update_strategy(size_t action_offset,
                       const double* action_probabilities,
                       size_t action_count,
                       double reach_prob);
  std::optional<SampledChanceTransition> sample_chance_transition(
      uint32_t public_state_id,
      const CompactPublicState& state,
      CardMask known_private_cards);
  SampledFrozenChanceTransition sample_frozen_chance_transition(
      const PublicStateRow& row,
      const ExactBoardState& exact_board,
      CardMask known_private_cards);
  double chance_sampling_cfr(
      uint32_t public_state_id,
      const CompactPublicState& state,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range);
  double chance_sampling_frozen_regret_only(
      uint32_t public_state_id,
      const ExactBoardState& exact_board,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int update_player,
      int depth,
      bool use_atomic_updates);
};

} // namespace poker
