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
#include "absl/types/span.h"
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
    int64_t betting_history_transition_hits = 0;
    int64_t betting_history_transition_misses = 0;
  };

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  using PrivateBucketId = uint16_t;

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
  size_t get_info_set_count() const { return info_set_count_; }
  size_t get_tree_node_count() const {
    return public_state_rows_.empty() ? game_tree_->node_count()
                                      : public_state_rows_.size();
  }
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

  struct BettingHistoryKey {
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
    int history_size = 0;
    std::array<int, kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;

    bool operator==(const BettingHistoryKey& other) const;
  };

  struct BettingHistoryKeyHash {
    size_t operator()(const BettingHistoryKey& key) const;
  };

  using PublicBucketId = uint64_t;

  struct PublicStateKey {
    uint32_t betting_history_id = 0;
    PublicBucketId public_bucket = 0;

    bool operator==(const PublicStateKey& other) const;
  };

  struct PublicStateKeyHash {
    size_t operator()(const PublicStateKey& key) const;
  };

  struct BettingHistoryRow {
    BettingHistoryRow() {
      action_ids.fill(0);
      action_child_ids.fill(GameTree::Node::kInvalidBettingHistoryId);
    }

    int street = 0;
    int pot = 0;
    std::array<int, 2> stack = {0, 0};
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    std::array<int, 2> player_contributions = {0, 0};
    uint8_t action_count = 0;
    std::array<int, GameTree::kMaxActionsPerNode> action_ids;
    std::array<uint32_t, GameTree::kMaxActionsPerNode> action_child_ids;
    uint32_t chance_child_id = GameTree::Node::kInvalidBettingHistoryId;
  };

  struct IdentityCardAbstraction {
    PublicBucketId public_bucket(const GameState& state) const {
      return state.board_mask;
    }

    PrivateBucketId private_bucket(ComboId combo_id, const GameState&) const {
      return combo_id;
    }

    uint32_t private_bucket_count(const GameState&) const {
      return kComboCount;
    }
  };

  struct InfoSetRow {
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct InfoSetAddress {
    uint32_t public_state_id = 0;
    int player = 0;
    PrivateBucketId private_bucket = 0;
  };

  struct PublicStateRow {
    PublicStateRow() {
      action_ids.fill(0);
      action_child_ids.fill(GameTree::Node::kInvalidPublicStateId);
    }

    GameState state;
    uint32_t betting_history_id = GameTree::Node::kInvalidBettingHistoryId;
    PublicBucketId public_bucket = 0;
    bool is_terminal = false;
    bool is_chance_node = false;
    int player_to_act = -1;
    uint8_t action_count = 0;
    std::array<GameAction, GameTree::kMaxActionsPerNode> actions = {};
    std::array<int, GameTree::kMaxActionsPerNode> action_ids = {};
    std::array<uint32_t, GameTree::kMaxActionsPerNode> action_child_ids = {};
  };

  static constexpr int kPrivateBucketChunkSize = 64;
  static constexpr int kPrivateBucketChunkCount =
      (kComboCount + kPrivateBucketChunkSize - 1) / kPrivateBucketChunkSize;

  struct PrivateRowChunk {
    PrivateRowChunk() { rows.fill(-1); }

    std::array<int32_t, kPrivateBucketChunkSize> rows;
  };

  struct PublicInfoSetSlabPlayer {
    std::array<std::unique_ptr<PrivateRowChunk>, kPrivateBucketChunkCount>
        private_row_chunks;
    std::vector<InfoSetRow> rows;
  };

  struct PublicInfoSetSlab {
    std::array<PublicInfoSetSlabPlayer, kPlayerCount> players;
  };

  struct StrategyTablesView {
    const absl::flat_hash_map<BettingHistoryKey, uint32_t,
                              BettingHistoryKeyHash>*
        betting_history_ids = nullptr;
    const std::vector<BettingHistoryRow>* betting_history_rows = nullptr;
    const absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>*
        public_state_ids = nullptr;
    const std::vector<PublicStateRow>* public_state_rows = nullptr;
    const absl::flat_hash_map<uint64_t, uint32_t>* public_chance_child_ids =
        nullptr;
    const std::vector<std::unique_ptr<PublicInfoSetSlab>>*
        public_info_set_slabs = nullptr;
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
  // Set to true after warmup; blocks info-set allocation so the parallel
  // training phase only writes to the atomic regret/strategy arrays. Shared
  // worker solvers also receive read-only public-state rows via
  // StrategyTablesView.
  bool frozen_ = false;
  
  absl::flat_hash_map<BettingHistoryKey, uint32_t, BettingHistoryKeyHash>
      betting_history_ids_;
  absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>
      public_state_ids_;
  std::vector<PublicStateRow> public_state_rows_;
  absl::flat_hash_map<uint64_t, uint32_t> public_chance_child_ids_;
  std::vector<BettingHistoryRow> betting_history_rows_;
  size_t info_set_count_ = 0;
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
                                uint32_t root_public_state_id,
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
      GameTree::Node& node,
      int player,
      const PrivateCards& private_cards,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      uint32_t public_state_id,
      const PublicStateRow& row,
      int player,
      const PrivateCards& private_cards,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const InfoSetRow& row,
      const GameTree::Node& node,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const InfoSetRow& info_set_row,
      const PublicStateRow& public_state_row,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  void condition_ranges_for_actions(
      const TrainingRangeView& range,
      const GameState& state,
      uint32_t public_state_id,
      int player,
      const ActionChoices& action_choices,
      ConditionedRanges& conditioned_ranges);
  BettingHistoryKey make_betting_history_key(const GameState& state) const;
  BettingHistoryRow make_betting_history_row(const GameState& state) const;
  PublicStateKey make_public_state_key(uint32_t betting_history_id,
                                       const GameState& state) const;
  uint32_t get_or_create_betting_history_id(const GameState& state);
  uint32_t get_or_create_betting_history_id(GameTree::Node& node);
  uint32_t get_or_create_public_state_id(uint32_t betting_history_id,
                                         const GameState& state);
  uint32_t get_or_create_public_state_id(const GameState& state);
  uint32_t get_or_create_public_state_id(GameTree::Node& node);
  uint32_t get_or_create_public_state_id(GameTree::Node& node,
                                         uint32_t betting_history_id);
  void cache_action_betting_history_transition(GameTree::Node& node,
                                               int action_index,
                                               GameTree::Node& child_node);
  void cache_chance_betting_history_transition(GameTree::Node& node,
                                               GameTree::Node& child_node);
  void cache_betting_history_actions(uint32_t betting_history_id,
                                     const GameTree::Node& node);
  void cache_betting_history_actions(uint32_t betting_history_id,
                                     const PublicStateRow& row);
  PublicStateRow make_public_state_row(uint32_t betting_history_id,
                                       const GameState& state) const;
  std::optional<uint32_t> get_or_create_public_state_row(
      uint32_t betting_history_id,
      const GameState& state);
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
  StrategyTablesView strategy_tables_view();
  std::optional<uint32_t> strategy_betting_history_id(
      const GameState& state) const;
  std::optional<uint32_t> strategy_betting_history_id(GameTree::Node& node);
  std::optional<uint32_t> strategy_public_state_id(GameTree::Node& node);
  const absl::flat_hash_map<BettingHistoryKey, uint32_t,
                            BettingHistoryKeyHash>&
  strategy_betting_history_ids() const;
  const std::vector<BettingHistoryRow>& strategy_betting_history_rows() const;
  const absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>&
  strategy_public_state_ids() const;
  const std::vector<PublicStateRow>& strategy_public_state_rows() const;
  const absl::flat_hash_map<uint64_t, uint32_t>&
  strategy_public_chance_child_ids() const;
  const std::vector<std::unique_ptr<PublicInfoSetSlab>>&
  strategy_public_info_set_slabs() const;
  const std::vector<int>& strategy_action_ids() const;
  const std::vector<float>& strategy_cumulative_regrets() const;
  const std::vector<float>& strategy_cumulative_strategies() const;
  std::vector<float>& mutable_strategy_cumulative_regrets();
  std::vector<float>& mutable_strategy_cumulative_strategies();
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
  double uncached_utility(const GameState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(GameTree::Node& node,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards);
  double evaluate_strategy_node(uint32_t public_state_id,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards);
  double evaluate_strategy_samples(
      int samples,
      uint32_t root_public_state_id,
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
  void update_strategy(const InfoSetRow& row,
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
