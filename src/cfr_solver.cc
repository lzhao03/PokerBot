#include "src/cfr_solver.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace poker {

namespace {

constexpr int kActionKeyMultiplier = 1000000;

Card MakeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

std::vector<Card> BuildDeck() {
  std::vector<Card> deck;
  deck.reserve(52);
  for (int suit = Suit::HEARTS; suit <= Suit::SPADES; ++suit) {
    for (int rank = 2; rank <= 14; ++rank) {
      deck.push_back(MakeCard(rank, static_cast<Suit>(suit)));
    }
  }
  return deck;
}

bool SameCard(const Card& left, const Card& right) {
  return left.rank() == right.rank() && left.suit() == right.suit();
}

void RemoveCard(std::vector<Card>* deck, const Card& card) {
  auto it = std::find_if(deck->begin(), deck->end(), [&](const Card& deck_card) {
    return SameCard(deck_card, card);
  });
  if (it != deck->end()) {
    deck->erase(it);
  }
}

void RemoveHand(std::vector<Card>* deck, const Hand& hand) {
  for (const Card& card : hand.cards()) {
    RemoveCard(deck, card);
  }
}

void RemoveBoard(std::vector<Card>* deck, const BoardState& state) {
  for (const Card& card : state.cards()) {
    RemoveCard(deck, card);
  }
}

int CardsForNextStreet(Street street) {
  switch (street) {
    case Street::PREFLOP:
      return 3;
    case Street::FLOP:
    case Street::TURN:
      return 1;
    case Street::RIVER:
      return 0;
    default:
      return 0;
  }
}

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

Hand DealHand(std::vector<Card>* deck) {
  if (deck->size() < 2) {
    throw std::runtime_error("Not enough cards to deal a hand");
  }

  Hand hand;
  *hand.add_cards() = deck->back();
  deck->pop_back();
  *hand.add_cards() = deck->back();
  deck->pop_back();
  return hand;
}

std::vector<Card> SampleStreetCards(
    const BoardState& state,
    const Hand& player_a_hand,
    const Hand& player_b_hand,
    std::mt19937* rng) {
  int count = CardsForNextStreet(state.street());
  std::vector<Card> deck = BuildDeck();
  RemoveHand(&deck, player_a_hand);
  RemoveHand(&deck, player_b_hand);
  RemoveBoard(&deck, state);

  if (deck.size() < static_cast<size_t>(count)) {
    throw std::runtime_error("Not enough cards to sample next street");
  }

  std::shuffle(deck.begin(), deck.end(), *rng);
  return std::vector<Card>(deck.end() - count, deck.end());
}

}  // namespace

CFRSolver::CFRSolver(const PokerConfig& config)
  : config_(config),
    game_tree_(new GameTree(config)),
    hand_evaluator_(new HandEvaluator()),
    info_set_abstraction_(new InfoSetAbstraction()),
    rng_(12345) {
}

CFRSolver::~CFRSolver() {
  delete game_tree_;
  delete hand_evaluator_;
  delete info_set_abstraction_;
}

void CFRSolver::run(int iterations) {
  // Initialize the game tree
  std::cout << "Building game tree..." << std::endl;
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
  
  GameTree::Node* root = game_tree_->build_tree(initial_state);
  
  std::cout << "Game tree built with " << root->legal_actions.size() << " legal actions at root" << std::endl;
  
  // Run iterations of CFR
  std::cout << "Starting CFR iterations..." << std::endl;
  for (int i = 0; i < iterations; ++i) {
    std::vector<Card> deck = BuildDeck();
    std::shuffle(deck.begin(), deck.end(), rng_);
    Hand player_a_hand = DealHand(&deck);
    Hand player_b_hand = DealHand(&deck);
    
    // Initialize reach probabilities
    std::vector<double> reach_probabilities(2, 1.0);
    
    // Run CFR algorithm with a maximum depth
    const int max_depth = config_.max_depth() > 0 ? config_.max_depth() : 3;
    std::cout << "Iteration " << i+1 << "/" << iterations << std::endl;
    cfr(root, player_a_hand, player_b_hand, reach_probabilities, i, 0, max_depth);
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
  // Check depth limit to prevent infinite recursion
  if (depth >= max_depth) {
    return 0.0;
  }
  
  // If the node is a terminal node, return the utility
  if (node->is_terminal) {
    return game_tree_->get_utility(node->state, player_a_hand, player_b_hand);
  }
  
  // If the node is a chance node, sample an outcome and continue
  if (node->is_chance_node) {
    return chance_sampling_cfr(node, player_a_hand, player_b_hand, reach_probabilities, iteration, depth, max_depth);
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
                    iteration, depth + 1, max_depth);
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

double CFRSolver::calculate_exploitability() const {
  // This is a placeholder - in a real implementation, this would compute
  // the exploitability of the current strategy
  return 0.0;
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
  
  // Clear the current strategy
  current_strategy_.clear();
  
  // Read the strategy from the file
  std::string line;
  std::string current_info_set;
  Strategy::ActionProbabilities current_action_probs;
  
  while (std::getline(file, line)) {
    if (line == "END_INFO_SET") {
      // End of an information set, update the strategy
      current_strategy_.update(current_info_set, current_action_probs);
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
  // This is a placeholder - in a real implementation, this would compute
  // the expected value of the game for the given player
  return 0.0;
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
