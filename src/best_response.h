#pragma once

#include "src/cfr_solver.h"
#include "src/hand_range.h"

namespace poker {

class BestResponseEvaluator {
 public:
  explicit BestResponseEvaluator(CFRSolver& solver) : solver_(solver) {}

  double calculate_exploitability();
  double calculate_exploitability(int samples);
  double calculate_exploitability(int samples,
                                  const HandRange& player_a_range,
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
  GameAction get_best_response_action(GameTree::Node& node,
                                      ComboId player_a_hand,
                                      ComboId player_b_hand,
                                      int best_response_player);

  double best_response_value(GameTree::Node& node,
                             ComboId player_a_hand,
                             ComboId player_b_hand,
                             int best_response_player);
  double best_response_value_against_range(
      GameTree::Node& node,
      ComboId best_response_hand,
      const WeightedHandRangeView& opponent_hands,
      int best_response_player);

 private:
  using StrategyProbabilities = absl::InlinedVector<double, 8>;

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
  void average_strategy_probabilities(
      GameTree::Node& node,
      int player,
      ComboId private_combo,
      StrategyProbabilities& probabilities);
  void average_strategy_probabilities(
      const StrategyTables::InfoSetRow& row,
      const GameTree::Node& node,
      double fallback_probability,
      StrategyProbabilities& probabilities);
  double utility(const GameState& state,
                 ComboId player_a_hand,
                 ComboId player_b_hand);

  CFRSolver& solver_;
};

}  // namespace poker
