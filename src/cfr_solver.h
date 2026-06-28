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

class CFRSolver {
public:
  CFRSolver(const PokerConfig& config);
  ~CFRSolver();
  
  // Run CFR for a specified number of iterations
  void run(int iterations);
  
  // The core CFR algorithm
  // Returns the expected value of the game for the current player
  double cfr(GameTree::Node* node, 
             const Hand& player_a_hand, 
             const Hand& player_b_hand,
             std::vector<double>& reach_probabilities, 
             int iteration,
             int depth = 0,
             int max_depth = 10);
  
  // Get the computed equilibrium strategy
  Strategy get_equilibrium_strategy() const;
  
  // Calculate the exploitability of the current strategy
  double calculate_exploitability() const;
  
  // Save and load the computed strategy
  void save_strategy(const std::string& filename) const;
  void load_strategy(const std::string& filename);
  
  // Get the expected value of the game for a player
  double get_expected_value(int player_id) const;

private:
  PokerConfig config_;
  GameTree* game_tree_;
  HandEvaluator* hand_evaluator_;
  InfoSetAbstraction* info_set_abstraction_;
  std::mt19937 rng_;
  
  // Regret tracking for each information set and action
  std::unordered_map<std::string, std::unordered_map<int, double>> cumulative_regrets_;
  
  // Strategy tracking for each information set and action
  std::unordered_map<std::string, std::unordered_map<int, double>> cumulative_strategy_;
  
  // Current iteration strategy
  Strategy current_strategy_;
  
  // Helper methods
  Strategy::ActionProbabilities get_strategy(
      const std::string& info_set_key,
      const std::vector<Action>& legal_actions);
  void update_strategy(const std::string& info_set_key, const Strategy::ActionProbabilities& strategy, double reach_prob);
  double chance_sampling_cfr(GameTree::Node* node, const Hand& player_a_hand, const Hand& player_b_hand, std::vector<double>& reach_probabilities, int iteration, int depth, int max_depth);
  double external_sampling_cfr(GameTree::Node* node, const Hand& player_a_hand, const Hand& player_b_hand, std::vector<double>& reach_probabilities, int iteration);
  double outcome_sampling_cfr(GameTree::Node* node, const Hand& player_a_hand, const Hand& player_b_hand, std::vector<double>& reach_probabilities, double sample_prob, int iteration);
};

} // namespace poker
