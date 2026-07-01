#pragma once

#include <random>
#include <unordered_map>
#include <string>
#include <vector>
#include "src/poker.pb.h"
#include "src/game_tree.h"
#include "src/strategy.h"
#include "src/info_set.h"
#include "src/hand_evaluator.h"

namespace poker {

class HandRange;

class CFRSolver {
public:
  CFRSolver(const PokerConfig& config);
  ~CFRSolver();
  
  // Run CFR for a specified number of iterations
  void run(int iterations);
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

private:
  friend class CFRSolverRegretTestPeer;

  PokerConfig config_;
  GameTree* game_tree_;
  HandEvaluator* hand_evaluator_;
  InfoSetAbstraction* info_set_abstraction_;
  std::mt19937 rng_;
  double cumulative_root_utility_;
  int iterations_run_;
  
  // CFR+ clipped regret tracking for each information set and action.
  std::unordered_map<std::string, std::unordered_map<int, double>> cumulative_regrets_;
  
  // Strategy tracking for each information set and action
  std::unordered_map<std::string, std::unordered_map<int, double>> cumulative_strategy_;
  
  // Current iteration strategy
  Strategy current_strategy_;
  
  // Helper methods
  GameTree::Node* get_or_build_root();
  void run_iterations(int iterations, const HandRange* player_a_range,
                      const HandRange* player_b_range, bool train_swapped);
  double evaluate_strategy_node(GameTree::Node* node,
                                const Hand& player_a_hand,
                                const Hand& player_b_hand,
                                const Strategy& strategy);
  double best_response_value(GameTree::Node* node,
                             const Hand& player_a_hand,
                             const Hand& player_b_hand,
                             const Strategy& strategy,
                             int best_response_player);
  Strategy::ActionProbabilities get_strategy(
      const std::string& info_set_key,
      const std::vector<Action>& legal_actions);
  void update_strategy(const std::string& info_set_key, const Strategy::ActionProbabilities& strategy, double reach_prob);
  double chance_sampling_cfr(GameTree::Node* node, const Hand& player_a_hand, const Hand& player_b_hand, std::vector<double>& reach_probabilities, int iteration, int depth, int max_depth);
};

} // namespace poker
