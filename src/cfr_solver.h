#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "src/poker.pb.h"
#include "src/game_tree.h"
#include "src/hand_range.h"
#include "src/strategy.h"
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

  CFRSolver(const PokerConfig& config);
  CFRSolver(const PokerConfig& config, const BoardState& initial_state);
  
  // Run CFR for a specified number of iterations
  void run(int iterations, const Hand& player_a_hand,
           const Hand& player_b_hand);
  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);
  
  // The core chance-sampled CFR+ algorithm.
  // Uses CFR+ regret clipping. Unless config.regret_only_training is set,
  // it also accumulates a linearly weighted average strategy.
  // Returns the expected value of the game for player A.
  // max_depth <= 0 disables the depth cutoff.
  double cfr(GameTree::Node& node,
             const Hand& player_a_hand, 
             const Hand& player_b_hand,
             std::array<double, 2>& reach_probabilities,
             int iteration,
             int depth = 0,
             int max_depth = 0);
  
  // Get the computed strategy. Regret-only training exports the current
  // regret-matched policy because average-strategy sums are not accumulated.
  Strategy get_equilibrium_strategy() const;
  double evaluate_strategy(const Hand& player_a_hand, const Hand& player_b_hand);
  double evaluate_strategy(int samples, const HandRange& player_a_range,
                           const HandRange& player_b_range);
  
  // Calculate sampled exploitability of the current strategy.
  double calculate_exploitability();
  double calculate_exploitability(int samples);
  double calculate_exploitability(int samples, const HandRange& player_a_range,
                                  const HandRange& player_b_range);
  double calculate_exploitability(const Hand& player_a_hand,
                                  const Hand& player_b_hand);
  double calculate_player_a_best_response_value(
      int samples,
      const HandRange& player_a_range,
      const HandRange& player_b_range);
  double calculate_player_b_best_response_value(
      int samples,
      const HandRange& player_a_range,
      const HandRange& player_b_range);
  // Debug helper for inspecting sampled best-response choices.
  Action get_best_response_action(GameTree::Node& node,
                                  const Hand& player_a_hand,
                                  const Hand& player_b_hand,
                                  int best_response_player);
  
  // Save and load the computed strategy
  void save_strategy(const std::string& filename) const;
  void load_strategy(const std::string& filename);
  
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
    std::discrete_distribution<size_t> player_a_distribution;
  };

  struct PrivateCards {
    static PrivateCards FromHand(const Hand& hand);
    static PrivateCards FromCombo(ComboId combo_id);

    CardMask mask() const;
    Hand to_hand() const;

    bool has_combo = false;
    ComboId combo = 0;
    Hand hand;
  };

  using OptionalTrainingRange =
      std::optional<std::reference_wrapper<const TrainingRangeView>>;

  struct ActionChoice {
    std::reference_wrapper<const Action> action;
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

  struct InfoSetKey {
    static constexpr int kMaxCards = 5;
    static constexpr int kInlineHistoryValues = 48;

    int player = 0;
    int street = 0;
    int pot = 0;
    int stack_a = 0;
    int stack_b = 0;
    int all_in = 0;
    int folded_player = 0;
    int player_to_act = 0;
    int player_contribution_size = 0;
    std::array<int, 2> player_contributions = {0, 0};
    int hand_size = 0;
    std::array<int, 2> hand_cards = {-1, -1};
    int board_size = 0;
    std::array<int, kMaxCards> board_cards = {-1, -1, -1, -1, -1};
    int history_size = 0;
    std::array<int, kInlineHistoryValues> history_values = {};
    std::vector<int> history_overflow;

    bool operator==(const InfoSetKey& other) const;
  };

  struct InfoSetKeyHash {
    size_t operator()(const InfoSetKey& key) const;
  };

  struct ActionState {
    int action_id = 0;
    float cumulative_regret = 0.0f;
    float cumulative_strategy = 0.0f;
  };

  struct InfoSetData {
    InfoSetKey key;
    std::vector<ActionState> actions;
  };

  CFRSolver(const PokerConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache);
  CFRSolver(const PokerConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache,
            std::shared_ptr<ContinuationValueProvider> continuation_value_provider);
  CFRSolver(const PokerConfig& config,
            std::shared_ptr<TerminalUtilityCache> utility_cache,
            std::shared_ptr<ContinuationValueProvider> continuation_value_provider,
            BoardState initial_state);

  PokerConfig config_;
  BoardState initial_state_;
  std::unique_ptr<GameTree> game_tree_;
  std::mt19937 rng_;
  double cumulative_root_utility_;
  int iterations_run_;
  int64_t cfr_update_count_;
  TraversalStats traversal_stats_;
  std::shared_ptr<TerminalUtilityCache> utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider_;
  
  absl::flat_hash_map<InfoSetKey, int, InfoSetKeyHash> info_set_ids_;
  std::vector<InfoSetData> info_sets_;
  
  // String-keyed strategy loaded from snapshots. Trained CFR state lives in
  // info_sets_ above.
  Strategy loaded_strategy_;
  
  // Helper methods
  GameTree::Node& get_or_build_root();
  void run_iterations(int iterations,
                      const HandRange& player_a_range,
                      const HandRange& player_b_range);
  double cfr_with_ranges(
      GameTree::Node& node,
      const Hand& player_a_hand,
      const Hand& player_b_hand,
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
  double average_strategy_action_probability(
      const BoardState& state,
      int player,
      const Hand& hand,
      const std::vector<Action>& legal_actions,
      int action_id);
  double average_strategy_action_probability(
      const BoardState& state,
      int player,
      const PrivateCards& private_cards,
      const std::vector<Action>& legal_actions,
      int action_id);
  double average_strategy_action_probability(
      const InfoSetData& info_set,
      const std::vector<Action>& legal_actions,
      int action_id,
      double fallback_probability);
  void average_strategy_probabilities(
      const BoardState& state,
      int player,
      const PrivateCards& private_cards,
      const std::vector<Action>& legal_actions,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const InfoSetData& info_set,
      const std::vector<Action>& legal_actions,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  void condition_ranges_for_actions(
      const TrainingRangeView& range,
      const BoardState& state,
      int player,
      const ActionChoices& action_choices,
      ConditionedRanges& conditioned_ranges);
  InfoSetKey make_info_set_key(const BoardState& state,
                               int player,
                               const Hand& hand) const;
  InfoSetKey make_info_set_key(const BoardState& state,
                               int player,
                               ComboId combo_id) const;
  InfoSetKey make_info_set_key(const BoardState& state,
                               int player,
                               const PrivateCards& private_cards) const;
  InfoSetKey make_public_info_set_key(const BoardState& state,
                                      int player) const;
  int get_or_create_info_set_id(const InfoSetKey& key,
                                const std::vector<int>& legal_action_ids);
  void initialize_info_set_actions(InfoSetData& info_set,
                                   const std::vector<int>& legal_action_ids);
  std::string info_set_key_to_string(const InfoSetKey& key) const;
  double regret_for_info_set(const std::string& info_set_key,
                             int action_id) const;
  ContinuationContext build_continuation_context(
      const BoardState& state,
      const Hand& player_a_hand,
      const Hand& player_b_hand,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range) const;
  double utility(const BoardState& state,
                 const Hand& player_a_hand,
                 const Hand& player_b_hand);
  double utility(const BoardState& state,
                 const PrivateCards& player_a_cards,
                 const PrivateCards& player_b_cards);
  double uncached_utility(const BoardState& state,
                          const PrivateCards& player_a_cards,
                          const PrivateCards& player_b_cards);
  double evaluate_strategy_node(GameTree::Node& node,
                                const Hand& player_a_hand,
                                const Hand& player_b_hand);
  double evaluate_strategy_node(GameTree::Node& node,
                                const PrivateCards& player_a_cards,
                                const PrivateCards& player_b_cards);
  double evaluate_strategy_samples(
      int samples,
      RangeSampler range_sampler);
  double best_response_value(GameTree::Node& node,
                             const Hand& player_a_hand,
                             const Hand& player_b_hand,
                             int best_response_player);
  double best_response_value_against_range(
      GameTree::Node& node,
      const Hand& best_response_hand,
      const WeightedHandRangeView& opponent_hands,
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
      const Hand& player_a_hand,
      const Hand& player_b_hand,
      std::array<double, 2>& reach_probabilities,
      int iteration,
      int depth,
      int max_depth,
      TraversalScratch& scratch,
      OptionalTrainingRange player_a_range,
      OptionalTrainingRange player_b_range);
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
