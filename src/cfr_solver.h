#pragma once

#include <array>
#include <atomic>
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
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/poker_types.h"
#include "src/strategy_tables.h"
#include "src/training_range.h"

namespace poker {

struct ContinuationContext;
class ContinuationValueProvider;
class TerminalUtilityCache;
class BestResponseEvaluator;

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
    int64_t betting_history_transition_hits = 0;
    int64_t betting_history_transition_misses = 0;
  };

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  using PrivateBucketId = StrategyTables::PrivateBucketId;

  struct StrategyInfoSetKey {
    uint32_t public_state_id = 0;
    PrivateBucketId private_bucket = 0;
    int player = 0;
  };

  struct StrategyInfoSet {
    StrategyInfoSetKey key;
    std::vector<int> action_ids;
    std::vector<double> probabilities;
  };

  struct StrategyProfile {
    std::vector<StrategyInfoSet> info_sets;

    bool empty() const { return info_sets.empty(); }
    size_t size() const { return info_sets.size(); }
  };

  CFRSolver(const SolverConfig& config);
  CFRSolver(const SolverConfig& config, const GameState& initial_state);
  
  // Run CFR for a specified number of iterations
  void run(int iterations, ComboId player_a_hand, ComboId player_b_hand);
  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);
  
  // Get the computed strategy. Regret-only training exports the current
  // regret-matched policy because average-strategy sums are not accumulated.
  StrategyProfile get_strategy_profile() const;
  double evaluate_strategy(ComboId player_a_hand, ComboId player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);
  
  // Calculate sampled exploitability of the current strategy.
  double calculate_exploitability();
  double calculate_exploitability(int samples);
  double calculate_exploitability(int samples, const HandRange& player_a_range,
                                  const HandRange& player_b_range);
  double calculate_exploitability(ComboId player_a_hand,
                                  ComboId player_b_hand);
  double calculate_player_a_best_response_value(
      int samples,
      const HandRange& player_a_range,
      const HandRange& player_b_range);
  double calculate_player_b_best_response_value(
      int samples,
      const HandRange& player_a_range,
      const HandRange& player_b_range);
  // Debug helper for inspecting sampled best-response choices.
  GameAction get_best_response_action(GameTree::Node& node,
                                      ComboId player_a_hand,
                                      ComboId player_b_hand,
                                      int best_response_player);
  
  // Get the expected value of the game for a player
  double get_expected_value(int player_id) const;
  int get_iterations_run() const { return iterations_run_.load(std::memory_order_relaxed); }
  int64_t get_cfr_update_count() const { return cfr_update_count_.load(std::memory_order_relaxed); }
  size_t get_info_set_count() const { return tables_->info_set_count; }
  size_t get_public_state_count() const {
    return tables_->public_state_rows.size();
  }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  void add_traversal_stats(const TraversalStats& stats);
  UtilityCacheStats get_utility_cache_stats() const;
  void set_continuation_value_provider(
      std::shared_ptr<ContinuationValueProvider> provider);

private:
  friend class BestResponseEvaluator;
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

  using BettingHistoryKey = StrategyTables::BettingHistoryKey;
  using BettingHistoryKeyHash = StrategyTables::BettingHistoryKeyHash;
  using PublicBucketId = StrategyTables::PublicBucketId;
  using PublicStateKey = StrategyTables::PublicStateKey;
  using PublicStateKeyHash = StrategyTables::PublicStateKeyHash;
  using BettingHistoryRow = StrategyTables::BettingHistoryRow;

  struct IdentityCardAbstraction {
    template <typename State>
    PublicBucketId public_bucket(const State& state) const {
      return state.board_mask;
    }

    template <typename State>
    PrivateBucketId private_bucket(ComboId combo_id, const State&) const {
      return combo_id;
    }

    template <typename State>
    uint32_t private_bucket_count(const State&) const {
      return kComboCount;
    }
  };

  using InfoSetRow = StrategyTables::InfoSetRow;
  using InfoSetAddress = StrategyTables::InfoSetAddress;
  using PublicStateRow = StrategyTables::PublicStateRow;
  using PrivateRowChunk = StrategyTables::PrivateRowChunk;
  using PublicInfoSetSlabPlayer = StrategyTables::PublicInfoSetSlabPlayer;
  using PublicInfoSetSlab = StrategyTables::PublicInfoSetSlab;
  static constexpr int kPrivateBucketChunkSize =
      StrategyTables::kPrivateBucketChunkSize;

  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache);
  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache,
            std::shared_ptr<ContinuationValueProvider> continuation_value_provider);
  CFRSolver(const SolverConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache,
            std::shared_ptr<ContinuationValueProvider> continuation_value_provider,
            GameState initial_state);

  SolverConfig config_;
  GameState initial_state_;
  std::shared_ptr<GameTree> game_tree_;
  std::mt19937 rng_;
  double cumulative_root_utility_;
  std::atomic<int> iterations_run_{0};
  std::atomic<int64_t> cfr_update_count_{0};
  TraversalStats traversal_stats_;
  std::shared_ptr<TerminalUtilityCache> utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider_;
  IdentityCardAbstraction card_abstraction_;
  // Set to true after warmup; blocks info-set allocation so the parallel
  // training phase only writes to the shared regret/strategy arrays.
  bool frozen_ = false;
  std::shared_ptr<StrategyTables> tables_;
  
  // Helper methods
  GameTree::Node& get_or_build_root();
  void run_iterations(int iterations,
                      const HandRange& player_a_range,
                      const HandRange& player_b_range);
  void run_iterations_parallel(int iterations,
                                int num_threads,
                                uint32_t root_public_state_id,
                                const RangeSampler& range_sampler,
                                const TrainingRange& player_a_training_range,
                                const TrainingRange& player_b_training_range);
  double cfr_with_ranges(
      uint32_t public_state_id,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range);
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
  void validate_public_state_row_actions(uint32_t public_state_id) const;
  BettingHistoryKey make_betting_history_key(const GameState& state) const;
  BettingHistoryKey make_betting_history_key(
      const CompactPublicState& state) const;
  BettingHistoryRow make_betting_history_row(const GameState& state) const;
  BettingHistoryRow make_betting_history_row(
      const CompactPublicState& state) const;
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const GameState& state) const;
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const CompactPublicState& state) const;
  uint32_t get_or_create_betting_history_id(const GameState& state);
  uint32_t get_or_create_betting_history_id(
      const CompactPublicState& state);
  uint32_t get_or_create_betting_history_id(GameTree::Node& node);
  uint32_t get_or_create_action_child_betting_history_id(
      uint32_t parent_betting_history_id,
      int action_index,
      const CompactPublicState& child_state);
  uint32_t get_or_create_chance_child_betting_history_id(
      uint32_t parent_betting_history_id,
      const CompactPublicState& child_state);
  void cache_action_betting_history_transition(GameTree::Node& node,
                                               int action_index,
                                               GameTree::Node& child_node);
  void cache_chance_betting_history_transition(GameTree::Node& node,
                                               GameTree::Node& child_node);
  void cache_betting_history_actions(uint32_t betting_history_id,
                                     const GameTree::Node& node);
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
  std::optional<uint32_t> get_or_create_action_child_public_state(
      uint32_t public_state_id,
      int action_index);
  std::optional<uint32_t> get_or_create_chance_child_public_state(
      uint32_t public_state_id,
      absl::Span<const CardId> cards);
  std::optional<InfoSetRow> get_or_create_info_set_row(
      InfoSetAddress address,
      const int* action_ids,
      int num_actions);
  std::optional<uint32_t> strategy_betting_history_id(
      const GameState& state) const;
  std::optional<uint32_t> strategy_betting_history_id(GameTree::Node& node);
  std::optional<uint32_t> strategy_public_state_id(GameTree::Node& node);
  InfoSetRow append_info_set_actions(const int* action_ids, int num_actions);
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
  ContinuationContext build_continuation_context(
      const GameState& state,
      ComboId player_a_hand,
      ComboId player_b_hand,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range) const;
  double utility(const GameState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double utility(const CompactPublicState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double uncached_utility(const GameState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double uncached_utility(const CompactPublicState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(uint32_t public_state_id,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards);
  double evaluate_strategy_samples(
      int samples,
      uint32_t root_public_state_id,
      RangeSampler range_sampler);
  void update_strategy(const InfoSetRow& row,
                       const double* action_probabilities,
                       size_t action_count,
                       double reach_prob);
  double chance_sampling_cfr(
      uint32_t public_state_id,
      const PrivateCards& player_a_cards,
      const PrivateCards& player_b_cards,
      std::array<double, 2>& reach_probabilities,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range);
};

} // namespace poker
