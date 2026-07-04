#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <random>
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
    ComboId private_combo = 0;
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
  int get_iterations_run() const { return iterations_run_; }
  int64_t get_cfr_update_count() const { return cfr_update_count_; }
  size_t get_info_set_count() const { return info_sets_.size(); }
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
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

    RangeDeal sample(std::mt19937& rng);

    const TrainingRange& player_a_range;
    const TrainingRange& player_b_range;
    std::vector<float> compatible_player_b_weight;
    std::vector<float> player_a_sample_weights;
    std::vector<uint32_t> compatible_player_b_offsets;
    std::vector<uint16_t> compatible_player_b_counts;
    std::vector<ComboId> compatible_player_b_combos;
    std::vector<float> compatible_player_b_cumulative_weights;
    std::discrete_distribution<size_t> player_a_distribution;
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
  using ConditionedRanges = absl::InlinedVector<TrainingRangeView, 8>;
  using StrategyProbabilities = absl::InlinedVector<double, 8>;

  struct RangeScratchFrame {
    ConditionedRanges conditioned_ranges;
    TrainingRangeView public_player_a_range;
    TrainingRangeView public_player_b_range;
  };

  struct TraversalScratch {
    RangeScratchFrame& frame(size_t depth) {
      while (frames.size() <= depth) {
        frames.emplace_back();
      }
      return frames[depth];
    }

    std::deque<RangeScratchFrame> frames;
  };

  struct PublicStateKey {
    static constexpr int kMaxCards = 5;
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
    int board_size = 0;
    std::array<int, kMaxCards> board_cards = {-1, -1, -1, -1, -1};
    int history_size = 0;
    std::array<int, kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;

    bool operator==(const PublicStateKey& other) const;
  };

  struct PublicStateKeyHash {
    size_t operator()(const PublicStateKey& key) const;
  };

  struct CompactInfoSetKey {
    uint32_t public_state_id = 0;
    uint16_t private_combo = 0;
    uint8_t player = 0;

    bool operator==(const CompactInfoSetKey& other) const;
  };

  struct CompactInfoSetKeyHash {
    size_t operator()(const CompactInfoSetKey& key) const;
  };

  struct ComboInfoSetIndex {
    ComboInfoSetIndex() { info_set_ids.fill(-1); }

    std::array<int32_t, kComboCount> info_set_ids;
  };

  struct InfoSetData {
    uint32_t public_state_id = 0;
    ComboId private_combo = 0;
    uint8_t player = 0;
    uint32_t action_offset = 0;
    uint16_t action_count = 0;
  };

  struct StrategyTablesView {
    const absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>*
        public_state_ids = nullptr;
    const absl::flat_hash_map<CompactInfoSetKey, int, CompactInfoSetKeyHash>*
        compact_info_set_ids = nullptr;
    const std::vector<InfoSetData>* info_sets = nullptr;
    const std::vector<int>* action_ids = nullptr;
    const std::vector<float>* cumulative_regrets = nullptr;
    const std::vector<float>* cumulative_strategies = nullptr;
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
  std::unique_ptr<GameTree> game_tree_;
  std::mt19937 rng_;
  double cumulative_root_utility_;
  int iterations_run_;
  int64_t cfr_update_count_;
  TraversalStats traversal_stats_;
  std::shared_ptr<TerminalUtilityCache> utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider_;
  
  absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>
      public_state_ids_;
  std::vector<InfoSetData> info_sets_;
  std::vector<int> action_ids_;
  std::vector<float> cumulative_regrets_;
  std::vector<float> cumulative_strategies_;
  absl::flat_hash_map<CompactInfoSetKey, int, CompactInfoSetKeyHash>
      compact_info_set_ids_;
  std::vector<std::unique_ptr<ComboInfoSetIndex>> combo_info_set_indexes_;
  std::array<absl::flat_hash_map<uint32_t, int32_t>, 2>
      combo_info_set_index_ids_by_public_state_;
  const StrategyTablesView* strategy_tables_view_ = nullptr;
  
  // Helper methods
  GameTree::Node& get_or_build_root();
  void run_iterations(int iterations,
                      const HandRange& player_a_range,
                      const HandRange& player_b_range);
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
      const InfoSetData& info_set,
      const std::vector<GameAction>& legal_actions,
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
  uint32_t get_or_create_public_state_id(const GameState& state,
                                         uint32_t node_id);
  int get_or_create_compact_info_set_id(
      uint32_t public_state_id,
      GameTree::Node* node,
      int player,
      ComboId combo_id,
      const std::vector<int>& legal_action_ids);
  ComboInfoSetIndex& get_or_build_combo_info_set_index(
      GameTree::Node& node,
      int player,
      uint32_t public_state_id);
  ComboInfoSetIndex* combo_info_set_index(GameTree::Node* node,
                                          int player,
                                          uint32_t public_state_id);
  StrategyTablesView strategy_tables_view() const;
  const absl::flat_hash_map<PublicStateKey, uint32_t, PublicStateKeyHash>&
  strategy_public_state_ids() const;
  const absl::flat_hash_map<CompactInfoSetKey, int, CompactInfoSetKeyHash>&
  strategy_compact_info_set_ids() const;
  const std::vector<InfoSetData>& strategy_info_sets() const;
  const std::vector<int>& strategy_action_ids() const;
  const std::vector<float>& strategy_cumulative_regrets() const;
  const std::vector<float>& strategy_cumulative_strategies() const;
  void initialize_info_set_actions(InfoSetData& info_set,
                                   const std::vector<int>& legal_action_ids);
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
