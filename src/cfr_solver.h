#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/poker_types.h"
#include "src/training_range.h"

namespace poker {

struct ContinuationContext;
class ContinuationValueProvider;
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
  };

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  struct StrategyInfoSetKey {
    uint32_t public_state_id = 0;
    uint16_t private_id = 0;
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
  
  // The core chance-sampled CFR+ algorithm.
  // Uses CFR+ regret clipping. Unless config.regret_only_training is set,
  // it also accumulates a linearly weighted average strategy.
  // Returns the expected value of the game for player A.
  // max_depth <= 0 disables the depth cutoff.
  double cfr(GameTree::Node& node,
             ComboId player_a_hand,
             ComboId player_b_hand,
             std::array<double, 2>& reach_probabilities,
             int iteration,
             int depth = 0,
             int max_depth = 0);
  
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
  size_t get_info_set_count() const { return info_sets_.size(); }
  size_t get_tree_node_count() const { return game_tree_->node_count(); }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  void add_traversal_stats(const TraversalStats& stats);
  UtilityCacheStats get_utility_cache_stats() const;
  void set_continuation_value_provider(
      std::shared_ptr<ContinuationValueProvider> provider);

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

  using OptionalTrainingRange =
      std::optional<std::reference_wrapper<const TrainingRangeView>>;

  struct ActionChoice {
    std::reference_wrapper<const GameAction> action;
    int action_id = 0;
    double probability = 0.0;
    double value = 0.0;
  };
  using ActionChoices = absl::InlinedVector<ActionChoice, 8>;
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

  struct PublicStateKey {
    static constexpr int kInlineHistoryValues = 48;

    int street = 0;
    int pot = 0;
    int stack_a = 0;
    int stack_b = 0;
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    int player_contribution_size = 0;
    std::array<int, 2> player_contributions = {0, 0};
    uint64_t public_cards_id = 0;
    int history_size = 0;
    std::array<int, kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;

    bool operator==(const PublicStateKey& other) const;
  };

  struct PublicStateKeyHash {
    size_t operator()(const PublicStateKey& key) const;
  };

  struct IdentityCardAbstraction {
    uint64_t public_id(const GameState& state) const {
      return state.board_mask;
    }

    uint16_t private_id(ComboId combo_id, const GameState&) const {
      return combo_id;
    }

    uint32_t private_id_count(const GameState&) const {
      return kComboCount;
    }
  };

  struct InfoSetData {
    uint32_t public_state_id = 0;
    uint16_t private_id = 0;
    uint8_t player = 0;
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
    int32_t info_set_id = -1;
  };

  static constexpr int kPrivateIdChunkSize = 64;
  static constexpr int kPrivateIdChunkCount =
      (kComboCount + kPrivateIdChunkSize - 1) / kPrivateIdChunkSize;

  struct PrivateRowChunk {
    PrivateRowChunk() { rows.fill(-1); }

    std::array<int32_t, kPrivateIdChunkSize> rows;
  };

  struct PublicInfoSetSlabPlayer {
    std::array<std::unique_ptr<PrivateRowChunk>, kPrivateIdChunkCount>
        private_row_chunks;
    std::vector<InfoSetRow> rows;
  };

  struct PublicInfoSetSlab {
    std::array<PublicInfoSetSlabPlayer, kPlayerCount> players;
  };

  struct StrategyTablesView {
    const absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>*
        public_state_ids = nullptr;
    const std::vector<std::unique_ptr<PublicInfoSetSlab>>*
        public_info_set_slabs = nullptr;
    const std::vector<InfoSetData>* info_sets = nullptr;
    const std::vector<int>* action_ids = nullptr;
    // The cumulative arrays are shared read/write across worker threads.
    std::vector<float>* cumulative_regrets = nullptr;
    std::vector<float>* cumulative_strategies = nullptr;
  };

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
  // Set to true after warmup; blocks all tree/info-set allocation so the
  // parallel training phase only writes to the atomic regret/strategy arrays.
  bool frozen_ = false;
  
  absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>
      public_state_ids_;
  std::vector<InfoSetData> info_sets_;
  std::vector<int> action_ids_;
  std::vector<float> cumulative_regrets_;
  std::vector<float> cumulative_strategies_;
  std::vector<std::unique_ptr<PublicInfoSetSlab>> public_info_set_slabs_;
  const StrategyTablesView* strategy_tables_view_ = nullptr;
  
  // Helper methods
  GameTree::Node& get_or_build_root();
  void run_iterations(int iterations,
                      const HandRange& player_a_range,
                      const HandRange& player_b_range);
  void run_iterations_parallel(int iterations,
                                int num_threads,
                                const RangeSampler& range_sampler,
                                const TrainingRange& player_a_training_range,
                                const TrainingRange& player_b_training_range);
  double cfr_with_ranges(
      GameTree::Node& node,
      ComboId player_a_hand,
      ComboId player_b_hand,
      std::array<double, 2>& reach_probabilities,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range);
  double cfr_with_ranges(
      GameTree::Node& node,
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
      GameTree::Node& node,
      int player,
      const PrivateCards& private_cards,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const InfoSetRow& row,
      const GameTree::Node& node,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  void condition_ranges_for_actions(
      const TrainingRangeView& range,
      GameTree::Node& node,
      uint32_t public_state_id,
      int player,
      const ActionChoices& action_choices,
      ConditionedRanges& conditioned_ranges);
  PublicStateKey make_public_state_key(const GameState& state) const;
  uint32_t get_or_create_public_state_id(const GameState& state);
  uint32_t get_or_create_public_state_id(GameTree::Node& node);
  int get_or_create_info_set_id(
      uint32_t public_state_id,
      int player,
      uint16_t private_id,
      const int* action_ids,
      int num_actions);
  StrategyTablesView strategy_tables_view();
  std::optional<uint32_t> strategy_public_state_id(GameTree::Node& node);
  const absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>&
  strategy_public_state_ids() const;
  const std::vector<std::unique_ptr<PublicInfoSetSlab>>&
  strategy_public_info_set_slabs() const;
  const std::vector<InfoSetData>& strategy_info_sets() const;
  const std::vector<int>& strategy_action_ids() const;
  const std::vector<float>& strategy_cumulative_regrets() const;
  const std::vector<float>& strategy_cumulative_strategies() const;
  std::vector<float>& mutable_strategy_cumulative_regrets();
  std::vector<float>& mutable_strategy_cumulative_strategies();
  void initialize_info_set_actions(InfoSetData& info_set,
                                   const int* action_ids,
                                   int num_actions);
  PublicInfoSetSlab& get_or_create_public_info_set_slab(
      uint32_t public_state_id);
  const PublicInfoSetSlab* public_info_set_slab(
      uint32_t public_state_id) const;
  const InfoSetRow* find_info_set_row(uint32_t public_state_id,
                                      int player,
                                      uint16_t private_id) const;
  static const InfoSetRow* find_info_set_row(
      const PublicInfoSetSlabPlayer& player_slab,
      uint16_t private_id);
  static int32_t& get_or_create_private_row_slot(
      PublicInfoSetSlabPlayer& player_slab,
      uint16_t private_id);
  ContinuationContext build_continuation_context(
      const GameState& state,
      ComboId player_a_hand,
      ComboId player_b_hand,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range) const;
  double utility(const GameState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double uncached_utility(const GameState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(GameTree::Node& node,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards);
  double evaluate_strategy_samples(
      int samples,
      RangeSampler range_sampler);
  double best_response_value(GameTree::Node& node,
                             const PrivateCards& player_a_cards,
                             const PrivateCards& player_b_cards,
                             int best_response_player);
  double best_response_value_against_range(
      GameTree::Node& node,
      const PrivateCards& best_response_cards,
      const WeightedHandRangeView& opponent_hands,
      int best_response_player);
  double sampled_range_best_response_value(
      int samples,
      const HandRange& best_response_range,
      const HandRange& opponent_range,
      int best_response_player);
  double sampled_range_best_response_samples(
      int samples,
      const WeightedHandRange& best_response_hands,
      const WeightedHandRange& opponent_hands,
      int best_response_player);
  void update_strategy(int info_set_id,
                       const ActionChoices& choices,
                       double reach_prob);
  double chance_sampling_cfr(
      GameTree::Node& node,
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
