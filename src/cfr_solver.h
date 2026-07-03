#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include "src/poker.pb.h"
#include "src/game_tree.h"
#include "src/strategy.h"
#include "src/info_set.h"
#include "src/hand_evaluator.h"

namespace poker {

class HandRange;
struct ContinuationContext;
class ContinuationValueProvider;
class TerminalUtilityCache;

struct CanonicalPublicStateKey {
  int street = 0;
  int pot = 0;
  int stack_a = 0;
  int stack_b = 0;
  int all_in = 0;
  int folded_player = 0;
  int player_to_act = 0;
  std::array<int, 2> player_contributions = {0, 0};
  int board_size = 0;
  std::array<int, 5> board_cards = {-1, -1, -1, -1, -1};
  int history_bucket = 0;
  int last_player = -1;
  int last_action = 0;
  int last_amount = 0;

  bool operator==(const CanonicalPublicStateKey& other) const;
};

struct CanonicalPublicStateKeyHash {
  size_t operator()(const CanonicalPublicStateKey& key) const;
};

class CFRSolver {
public:
  struct TraversalStats {
    int64_t cfr_updates = 0;
    int64_t preflop_updates = 0;
    int64_t flop_updates = 0;
    int64_t turn_updates = 0;
    int64_t river_updates = 0;
    int max_decision_depth = 0;
    int64_t canonical_state_visits = 0;
    int64_t unique_canonical_states = 0;
    int64_t duplicate_canonical_state_visits = 0;
    int64_t child_nodes_created = 0;
    int64_t chance_samples = 0;
    int64_t terminal_utility_calls = 0;
    int64_t fold_utility_calls = 0;
    int64_t showdown_utility_calls = 0;
  };

  struct UtilityCacheStats {
    int64_t hits = 0;
    int64_t misses = 0;
    int64_t entries = 0;
  };

  CFRSolver(const PokerConfig& config);
  CFRSolver(const PokerConfig& config, const BoardState& initial_state);
  ~CFRSolver();
  
  // Run CFR for a specified number of iterations
  void run(int iterations);
  void run(int iterations, const Hand& player_a_hand,
           const Hand& player_b_hand);
  void run(int iterations, const HandRange& player_a_range,
           const HandRange& player_b_range);
  
  // The core chance-sampled CFR+ algorithm.
  // Uses CFR+ regret clipping and linearly weighted average strategy.
  // Returns the expected value of the game for player A.
  // max_depth <= 0 disables the depth cutoff.
  double cfr(GameTree::Node* node, 
             const Hand& player_a_hand, 
             const Hand& player_b_hand,
             std::vector<double>& reach_probabilities, 
             int iteration,
             int depth = 0,
             int max_depth = 0);
  
  // Get the computed equilibrium strategy
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
  Action get_best_response_action(GameTree::Node* node,
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
  TraversalStats get_traversal_stats() const { return traversal_stats_; }
  UtilityCacheStats get_utility_cache_stats() const;
  void set_continuation_value_provider(
      std::shared_ptr<ContinuationValueProvider> provider);

private:
  friend class CFRSolverRegretTestPeer;

  struct RangeDeal {
    Hand player_a_hand;
    Hand player_b_hand;
    double weight;
  };

  struct ActionChoice {
    const Action* action = nullptr;
    int action_id = 0;
    double probability = 0.0;
    double value = 0.0;
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
  GameTree* game_tree_;
  HandEvaluator* hand_evaluator_;
  InfoSetAbstraction* info_set_abstraction_;
  std::mt19937 rng_;
  double cumulative_root_utility_;
  int iterations_run_;
  int64_t cfr_update_count_;
  TraversalStats traversal_stats_;
  std::unordered_set<CanonicalPublicStateKey, CanonicalPublicStateKeyHash>
      visited_canonical_states_;
  std::shared_ptr<TerminalUtilityCache> utility_cache_;
  std::shared_ptr<ContinuationValueProvider> continuation_value_provider_;
  
  // CFR+ clipped regret tracking for each information set and action.
  std::unordered_map<std::string, std::unordered_map<int, double>> cumulative_regrets_;
  
  // Strategy tracking for each information set and action
  std::unordered_map<std::string, std::unordered_map<int, double>> cumulative_strategy_;
  
  // Current iteration strategy
  Strategy current_strategy_;
  
  // Helper methods
  GameTree::Node* get_or_build_root();
  static std::vector<RangeDeal> build_compatible_range_deals(
      const std::vector<std::pair<Hand, double>>& player_a_hands,
      const std::vector<std::pair<Hand, double>>& player_b_hands);
  void run_iterations(int iterations, const HandRange* player_a_range,
                      const HandRange* player_b_range, bool train_swapped);
  double cfr_with_ranges(
      GameTree::Node* node,
      const Hand& player_a_hand,
      const Hand& player_b_hand,
      std::vector<double>& reach_probabilities,
      int iteration,
      int depth,
      int max_depth,
      const std::vector<std::pair<Hand, double>>* player_a_range,
      const std::vector<std::pair<Hand, double>>* player_b_range);
  double action_probability_for_hand(
      const BoardState& state,
      int player,
      const Hand& hand,
      const std::vector<Action>& legal_actions,
      int action_id) const;
  std::vector<std::pair<Hand, double>> condition_range_for_action(
      const std::vector<std::pair<Hand, double>>& range,
      const BoardState& state,
      int player,
      const std::vector<Action>& legal_actions,
      int action_id) const;
  ContinuationContext build_continuation_context(
      const BoardState& state,
      const Hand& player_a_hand,
      const Hand& player_b_hand,
      const std::vector<std::pair<Hand, double>>* player_a_range,
      const std::vector<std::pair<Hand, double>>* player_b_range) const;
  double utility(const BoardState& state,
                 const Hand& player_a_hand,
                 const Hand& player_b_hand);
  double evaluate_strategy_node(GameTree::Node* node,
                                const Hand& player_a_hand,
                                const Hand& player_b_hand,
                                const Strategy& strategy);
  double evaluate_strategy_samples(
      int samples,
      const std::vector<RangeDeal>& range_deals,
      const std::vector<double>& range_deal_weights,
      const Strategy& strategy);
  double best_response_value(GameTree::Node* node,
                             const Hand& player_a_hand,
                             const Hand& player_b_hand,
                             const Strategy& strategy,
                             int best_response_player);
  double best_response_value_against_range(
      GameTree::Node* node,
      const Hand& best_response_hand,
      const std::vector<std::pair<Hand, double>>& opponent_hands,
      const Strategy& strategy,
      int best_response_player);
  double sampled_range_best_response_value(
      int samples,
      const HandRange& best_response_range,
      const HandRange& opponent_range,
      const Strategy& strategy,
      int best_response_player);
  double sampled_range_best_response_samples(
      int samples,
      const std::vector<std::pair<Hand, double>>& best_response_hands,
      const std::vector<std::pair<Hand, double>>& opponent_hands,
      const Strategy& strategy,
      int best_response_player);
  Strategy::ActionProbabilities get_strategy(
      const std::string& info_set_key,
      const std::vector<Action>& legal_actions);
  std::vector<ActionChoice> get_action_choices(
      const std::string& info_set_key,
      const std::vector<Action>& legal_actions);
  void update_strategy(const std::string& info_set_key, const Strategy::ActionProbabilities& strategy, double reach_prob);
  void update_strategy(const std::string& info_set_key,
                       const std::vector<ActionChoice>& choices,
                       double reach_prob);
  double chance_sampling_cfr(
      GameTree::Node* node,
      const Hand& player_a_hand,
      const Hand& player_b_hand,
      std::vector<double>& reach_probabilities,
      int iteration,
      int depth,
      int max_depth,
      const std::vector<std::pair<Hand, double>>* player_a_range,
      const std::vector<std::pair<Hand, double>>* player_b_range);
};

} // namespace poker
