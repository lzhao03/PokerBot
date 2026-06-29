#include "src/cfr_solver.h"
#include "src/card_utils.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace poker {

namespace {

constexpr int kActionKeyMultiplier = 1000000;

int ActionKey(const Action& action) {
  // ponytail: amounts are whole chips today; use a structured key if fractional chips matter.
  return static_cast<int>(action.action()) * kActionKeyMultiplier +
         static_cast<int>(std::lround(action.amount()));
}

std::string ActionTypeName(ActionType action_type) {
  switch (action_type) {
    case ActionType::FOLD:
      return "fold";
    case ActionType::CHECK:
      return "check";
    case ActionType::CALL:
      return "call";
    case ActionType::BET:
      return "bet";
    case ActionType::RAISE:
      return "raise";
    case ActionType::ALL_IN:
      return "all_in";
    default:
      return "unknown";
  }
}

ActionType ActionTypeFromName(const std::string& name) {
  if (name == "fold") return ActionType::FOLD;
  if (name == "check") return ActionType::CHECK;
  if (name == "call") return ActionType::CALL;
  if (name == "bet") return ActionType::BET;
  if (name == "raise") return ActionType::RAISE;
  if (name == "all_in") return ActionType::ALL_IN;
  return ActionType::NO_ACTION;
}

std::string ActionKeyToString(int action_key) {
  ActionType action_type =
      static_cast<ActionType>(action_key / kActionKeyMultiplier);
  int amount = action_key % kActionKeyMultiplier;
  std::string name = ActionTypeName(action_type);
  if (amount == 0) {
    return name;
  }
  return name + " " + std::to_string(amount);
}

int ActionKeyFromString(const std::string& name, int amount) {
  return static_cast<int>(ActionTypeFromName(name)) * kActionKeyMultiplier + amount;
}

}  // namespace

CFRSolver::CFRSolver(const PokerConfig& config)
  : config_(config),
    game_tree_(new GameTree(config)),
    hand_evaluator_(new HandEvaluator()),
    info_set_abstraction_(new InfoSetAbstraction()),
    rng_(12345),
    cumulative_root_utility_(0.0),
    iterations_run_(0) {
}

CFRSolver::~CFRSolver() {
  delete game_tree_;
  delete hand_evaluator_;
  delete info_set_abstraction_;
}

GameTree::Node* CFRSolver::get_or_build_root() {
  const int small_blind = config_.small_blind() > 0 ? config_.small_blind() : 1;
  const int big_blind = config_.big_blind() > 0 ? config_.big_blind() : 2;
  const int starting_stack = config_.starting_stack_size();

  BoardState initial_state;
  initial_state.set_stack_a(std::max(0, starting_stack - small_blind));
  initial_state.set_stack_b(std::max(0, starting_stack - big_blind));
  initial_state.set_pot(small_blind + big_blind);
  initial_state.set_folded_player(-1); // No player has folded
  initial_state.set_street(Street::PREFLOP);
  initial_state.set_all_in(false);
  initial_state.set_player_to_act(0); // Player 0 acts first
  
  // Add initial player contributions (blinds)
  initial_state.add_player_contribution(small_blind);
  initial_state.add_player_contribution(big_blind);

  GameTree::Node* root = game_tree_->get_root();
  if (root == nullptr) {
    return game_tree_->build_tree(initial_state);
  }
  return root;
}

void CFRSolver::run(int iterations) {
  std::cout << "Preparing game tree..." << std::endl;
  bool had_root = game_tree_->get_root() != nullptr;
  GameTree::Node* root = get_or_build_root();
  if (!had_root) {
    std::cout << "Game tree built with " << root->legal_actions.size()
              << " legal actions at root" << std::endl;
  } else {
    std::cout << "Reusing game tree with " << root->legal_actions.size()
              << " legal actions at root" << std::endl;
  }
  
  // Run iterations of CFR
  std::cout << "Starting CFR iterations..." << std::endl;
  for (int i = 0; i < iterations; ++i) {
    std::vector<Card> deck = BuildDeck();
    std::shuffle(deck.begin(), deck.end(), rng_);
    Hand player_a_hand = DealHand(&deck);
    Hand player_b_hand = DealHand(&deck);
    
    // Initialize reach probabilities
    std::vector<double> reach_probabilities(2, 1.0);
    
    const int max_depth = config_.max_depth();
    std::cout << "Iteration " << i+1 << "/" << iterations << std::endl;
    cumulative_root_utility_ +=
        cfr(root, player_a_hand, player_b_hand, reach_probabilities, i, 0, max_depth);
    ++iterations_run_;
  }
  
  std::cout << "CFR iterations completed" << std::endl;
}

double CFRSolver::cfr(GameTree::Node* node, 
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::vector<double>& reach_probabilities, 
                      int iteration,
                      int depth,
                      int max_depth) {
  // If the node is a terminal node, return the utility
  if (node->is_terminal) {
    return game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
  }

  // Chance card deals are not player decisions, so they do not consume CFR depth.
  if (node->is_chance_node) {
    return chance_sampling_cfr(node, player_a_hand, player_b_hand,
                               reach_probabilities, iteration, depth, max_depth);
  }

  // Check depth limit to prevent infinite recursion
  if (max_depth > 0 && depth >= max_depth) {
    if (game_tree_->is_betting_round_over(node->state)) {
      return game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
    }
    return 0.0;
  }
  
  // Get the player to act at this node
  int player = node->player_to_act;
  
  // Get the player's hand
  const Hand& player_hand = (player == 0) ? player_a_hand : player_b_hand;
  
  // Get the information set key for this player
  std::string info_set_key = info_set_abstraction_->state_to_info_set(node->state, player, player_hand);
  
  // Get the current strategy for this information set
  Strategy::ActionProbabilities strategy =
      get_strategy(info_set_key, node->legal_actions);
  
  // Initialize expected values for each action
  std::unordered_map<int, double> action_values;
  
  // Initialize the expected value for the player
  double node_value = 0.0;
  
  // For each action, recursively call CFR and compute the expected value
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    
    // Create child node for this action if it doesn't exist
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    
    // Get the child node for this action
    GameTree::Node* child_node = node->children[action_id];
    
    // Update reach probabilities for the recursive call
    std::vector<double> new_reach_probabilities = reach_probabilities;
    new_reach_probabilities[player] *= strategy[action_id];
    
    // Recursive call to get the expected value of this action
    double action_value = cfr(child_node, player_a_hand, player_b_hand, new_reach_probabilities, iteration, depth + 1, max_depth);
    
    // Store the action value
    action_values[action_id] = action_value;
    
    // Update the expected value of the node
    node_value += strategy[action_id] * action_value;
  }
  
  // Compute counterfactual regrets if this is not a chance player
  if (player == 0 || player == 1) {
    // Compute the counterfactual reach probability of the opponent
    double opponent_reach_prob = reach_probabilities[1 - player];
    
    // For each action, compute and accumulate the counterfactual regret
    for (const Action& action : node->legal_actions) {
      int action_id = ActionKey(action);
      
      // get_utility returns player A's utility; player B's regret uses the
      // opposite payoff in this zero-sum game.
      double utility_sign = player == 0 ? 1.0 : -1.0;
      double regret =
          opponent_reach_prob * utility_sign * (action_values[action_id] - node_value);
      
      // Accumulate the regret
      cumulative_regrets_[info_set_key][action_id] += regret;
    }
    
    // Update the strategy based on the reach probability
    update_strategy(info_set_key, strategy, reach_probabilities[player]);
  }
  
  return node_value;
}

double CFRSolver::chance_sampling_cfr(GameTree::Node* node, 
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::vector<double>& reach_probabilities, 
                      int iteration,
                      int depth,
                      int max_depth) {
  std::vector<Card> cards =
      SampleStreetCards(node->state, player_a_hand, player_b_hand, &rng_);
  GameTree::Node* child_node = game_tree_->create_chance_child_node(node, cards);
  double value = cfr(child_node, player_a_hand, player_b_hand, reach_probabilities,
                    iteration, depth, max_depth);
  delete child_node;
  return value;
}

double CFRSolver::external_sampling_cfr(GameTree::Node* node, 
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::vector<double>& reach_probabilities, 
                      int iteration) {
  // This is a placeholder for the external sampling CFR variant
  // In a real implementation, this would sample actions for the opponent
  return 0.0;
}

double CFRSolver::outcome_sampling_cfr(GameTree::Node* node, 
                      const Hand& player_a_hand, 
                      const Hand& player_b_hand,
                      std::vector<double>& reach_probabilities, 
                      double sample_prob, 
                      int iteration) {
  // This is a placeholder for the outcome sampling CFR variant
  // In a real implementation, this would sample a single path through the tree
  return 0.0;
}

Strategy CFRSolver::get_equilibrium_strategy() const {
  Strategy equilibrium_strategy;
  
  // For each information set
  for (const auto& info_set_pair : cumulative_strategy_) {
    const std::string& info_set_key = info_set_pair.first;
    const auto& cumulative_action_probs = info_set_pair.second;
    
    // Compute the sum of cumulative probabilities
    double sum = 0.0;
    for (const auto& action_prob : cumulative_action_probs) {
      sum += action_prob.second;
    }
    
    // Normalize the probabilities
    Strategy::ActionProbabilities normalized_strategy;
    if (sum > 0.0) {
      for (const auto& action_prob : cumulative_action_probs) {
        normalized_strategy[action_prob.first] = action_prob.second / sum;
      }
    } else {
      // If sum is 0, use uniform strategy
      double uniform_prob = 1.0 / cumulative_action_probs.size();
      for (const auto& action_prob : cumulative_action_probs) {
        normalized_strategy[action_prob.first] = uniform_prob;
      }
    }
    
    // Update the equilibrium strategy
    equilibrium_strategy.update(info_set_key, normalized_strategy);
  }
  
  return equilibrium_strategy;
}

double CFRSolver::evaluate_strategy(const Hand& player_a_hand,
                                    const Hand& player_b_hand) {
  Strategy strategy = get_equilibrium_strategy();
  return evaluate_strategy_node(get_or_build_root(), player_a_hand, player_b_hand,
                                strategy);
}

double CFRSolver::evaluate_strategy_node(GameTree::Node* node,
                                         const Hand& player_a_hand,
                                         const Hand& player_b_hand,
                                         const Strategy& strategy) {
  if (node->is_terminal) {
    return game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
  }
  if (node->is_chance_node) {
    std::vector<Card> cards =
        SampleStreetCards(node->state, player_a_hand, player_b_hand, &rng_);
    GameTree::Node* child_node = game_tree_->create_chance_child_node(node, cards);
    double value =
        evaluate_strategy_node(child_node, player_a_hand, player_b_hand, strategy);
    delete child_node;
    return value;
  }
  if (node->legal_actions.empty()) {
    return 0.0;
  }

  int player = node->player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const Hand& player_hand = player == 0 ? player_a_hand : player_b_hand;
  std::string info_set_key =
      info_set_abstraction_->state_to_info_set(node->state, player, player_hand);

  std::vector<int> action_ids;
  action_ids.reserve(node->legal_actions.size());
  double probability_sum = 0.0;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    action_ids.push_back(action_id);
    probability_sum += strategy.get_action_probability(info_set_key, action_id);
  }

  double value = 0.0;
  for (size_t i = 0; i < node->legal_actions.size(); ++i) {
    const Action& action = node->legal_actions[i];
    int action_id = action_ids[i];
    double probability = probability_sum > 0.0
                             ? strategy.get_action_probability(info_set_key, action_id) /
                                   probability_sum
                             : 1.0 / node->legal_actions.size();
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    value += probability * evaluate_strategy_node(
                              node->children[action_id], player_a_hand,
                              player_b_hand, strategy);
  }
  return value;
}

double CFRSolver::best_response_value(GameTree::Node* node,
                                      const Hand& player_a_hand,
                                      const Hand& player_b_hand,
                                      const Strategy& strategy,
                                      int best_response_player) {
  if (node->is_terminal) {
    double player_a_value =
        game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
    return best_response_player == 0 ? player_a_value : -player_a_value;
  }
  if (node->is_chance_node) {
    std::vector<Card> cards =
        SampleStreetCards(node->state, player_a_hand, player_b_hand, &rng_);
    GameTree::Node* child_node = game_tree_->create_chance_child_node(node, cards);
    double value = best_response_value(child_node, player_a_hand, player_b_hand,
                                       strategy, best_response_player);
    delete child_node;
    return value;
  }
  if (node->legal_actions.empty()) {
    return 0.0;
  }

  int player = node->player_to_act;
  if (player != 0 && player != 1) {
    return 0.0;
  }

  const Hand& player_hand = player == 0 ? player_a_hand : player_b_hand;
  std::string info_set_key =
      info_set_abstraction_->state_to_info_set(node->state, player, player_hand);

  std::vector<int> action_ids;
  action_ids.reserve(node->legal_actions.size());
  double probability_sum = 0.0;
  for (const Action& action : node->legal_actions) {
    int action_id = ActionKey(action);
    action_ids.push_back(action_id);
    probability_sum += strategy.get_action_probability(info_set_key, action_id);
  }

  if (player == best_response_player) {
    double value = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < node->legal_actions.size(); ++i) {
      const Action& action = node->legal_actions[i];
      int action_id = action_ids[i];
      if (node->children.find(action_id) == node->children.end()) {
        node->children[action_id] = game_tree_->create_child_node(node, action);
      }
      value = std::max(value, best_response_value(
                                  node->children[action_id], player_a_hand,
                                  player_b_hand, strategy, best_response_player));
    }
    return value;
  }

  double value = 0.0;
  for (size_t i = 0; i < node->legal_actions.size(); ++i) {
    const Action& action = node->legal_actions[i];
    int action_id = action_ids[i];
    double probability = probability_sum > 0.0
                             ? strategy.get_action_probability(info_set_key, action_id) /
                                   probability_sum
                             : 1.0 / node->legal_actions.size();
    if (node->children.find(action_id) == node->children.end()) {
      node->children[action_id] = game_tree_->create_child_node(node, action);
    }
    value += probability * best_response_value(
                               node->children[action_id], player_a_hand,
                               player_b_hand, strategy, best_response_player);
  }
  return value;
}

double CFRSolver::calculate_exploitability() {
  return calculate_exploitability(1);
}

double CFRSolver::calculate_exploitability(int samples) {
  if (samples <= 0) {
    return 0.0;
  }

  double total = 0.0;
  for (int i = 0; i < samples; ++i) {
    std::vector<Card> deck = BuildDeck();
    std::shuffle(deck.begin(), deck.end(), rng_);
    Hand player_a_hand = DealHand(&deck);
    Hand player_b_hand = DealHand(&deck);
    total += calculate_exploitability(player_a_hand, player_b_hand);
  }
  return total / samples;
}

double CFRSolver::calculate_exploitability(const Hand& player_a_hand,
                                           const Hand& player_b_hand) {
  Strategy strategy = get_equilibrium_strategy();
  GameTree::Node* root = get_or_build_root();
  double strategy_player_a_value =
      evaluate_strategy_node(root, player_a_hand, player_b_hand, strategy);
  double player_a_gap =
      best_response_value(root, player_a_hand, player_b_hand, strategy, 0) -
      strategy_player_a_value;
  double player_b_gap =
      best_response_value(root, player_a_hand, player_b_hand, strategy, 1) +
      strategy_player_a_value;
  return (std::max(0.0, player_a_gap) + std::max(0.0, player_b_gap)) / 2.0;
}

void CFRSolver::save_strategy(const std::string& filename) const {
  std::ofstream file(filename);
  
  if (!file.is_open()) {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }
  
  // Get the equilibrium strategy
  Strategy equilibrium_strategy = get_equilibrium_strategy();
  
  // Write the strategy to the file
  for (const std::string& info_set_key : equilibrium_strategy.get_info_sets()) {
    file << info_set_key << "\n";
    
    Strategy::ActionProbabilities action_probs = equilibrium_strategy.get_strategy(info_set_key);
    for (const auto& action_prob : action_probs) {
      file << ActionKeyToString(action_prob.first) << " " << action_prob.second << "\n";
    }
    
    file << "END_INFO_SET\n";
  }
  
  file.close();
}

void CFRSolver::load_strategy(const std::string& filename) {
  std::ifstream file(filename);
  
  if (!file.is_open()) {
    std::cerr << "Failed to open file for reading: " << filename << std::endl;
    return;
  }
  
  current_strategy_.clear();
  cumulative_strategy_.clear();
  
  // Read the strategy from the file
  std::string line;
  std::string current_info_set;
  Strategy::ActionProbabilities current_action_probs;
  
  while (std::getline(file, line)) {
    if (line == "END_INFO_SET") {
      // End of an information set, update the strategy
      current_strategy_.update(current_info_set, current_action_probs);
      cumulative_strategy_[current_info_set] = current_action_probs;
      current_info_set.clear();
      current_action_probs.clear();
    } else if (current_info_set.empty()) {
      // This is an information set key
      current_info_set = line;
    } else {
      // This is an action probability
      std::istringstream iss(line);
      std::string action_name;
      int amount = 0;
      
      if (iss >> action_name) {
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
          tokens.push_back(token);
        }

        if (tokens.size() == 1) {
          current_action_probs[ActionKeyFromString(action_name, amount)] =
              std::stod(tokens[0]);
        } else if (tokens.size() == 2) {
          amount = std::stoi(tokens[0]);
          current_action_probs[ActionKeyFromString(action_name, amount)] =
              std::stod(tokens[1]);
        }
      }
    }
  }
  
  file.close();
}

double CFRSolver::get_expected_value(int player_id) const {
  if (iterations_run_ == 0) {
    return 0.0;
  }
  double player_a_ev = cumulative_root_utility_ / iterations_run_;
  return player_id == 0 ? player_a_ev : -player_a_ev;
}

Strategy::ActionProbabilities CFRSolver::get_strategy(
    const std::string& info_set_key,
    const std::vector<Action>& legal_actions) {
  Strategy::ActionProbabilities strategy;
  auto& regrets = cumulative_regrets_[info_set_key];
  if (legal_actions.empty()) {
    return strategy;
  }

  std::vector<int> action_ids;
  action_ids.reserve(legal_actions.size());
  for (const Action& action : legal_actions) {
    int action_id = ActionKey(action);
    if (regrets.find(action_id) == regrets.end()) {
      regrets[action_id] = 0.0;
    }
    if (std::find(action_ids.begin(), action_ids.end(), action_id) ==
        action_ids.end()) {
      action_ids.push_back(action_id);
    }
  }
  
  // Compute the sum of positive regrets
  double sum_positive_regrets = 0.0;
  for (int action_id : action_ids) {
    sum_positive_regrets += std::max(0.0, regrets[action_id]);
  }
  
  // If there are positive regrets, use regret matching
  if (sum_positive_regrets > 0.0) {
    for (int action_id : action_ids) {
      strategy[action_id] = std::max(0.0, regrets[action_id]) / sum_positive_regrets;
    }
  } else {
    // If all regrets are negative or zero, use a uniform strategy
    double uniform_prob = 1.0 / action_ids.size();
    for (int action_id : action_ids) {
      strategy[action_id] = uniform_prob;
    }
  }
  
  return strategy;
}

void CFRSolver::update_strategy(const std::string& info_set_key, const Strategy::ActionProbabilities& strategy, double reach_prob) {
  // Accumulate the strategy weighted by the reach probability
  for (const auto& action_prob : strategy) {
    cumulative_strategy_[info_set_key][action_prob.first] += reach_prob * action_prob.second;
  }
}

} // namespace poker
