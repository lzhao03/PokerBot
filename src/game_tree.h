#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include "src/hand_evaluator.h"
#include "src/poker.pb.h"

namespace poker {

class GameTree {
public:
  // Node in the game tree representing a decision point
  struct Node {
    BoardState state;
    bool is_terminal;
    bool is_chance_node;
    int player_to_act; // 0 for player A, 1 for player B, -1 for chance
    std::vector<Action> legal_actions;
    std::unordered_map<int, std::unique_ptr<Node>> children; // Action ID -> Node
    
    // For terminal nodes
    double utility;
    
    Node() : is_terminal(false), is_chance_node(false), player_to_act(-1), utility(0.0) {}
  };
  
  // Constructor
  GameTree(const PokerConfig& config);
  
  // Build the root node from the initial state
  Node& build_tree(const BoardState& initial_state);
  
  // Get the root node of the tree
  bool has_root() const { return root_ != nullptr; }
  Node& root();
  const Node& root() const;
  
  // Create a child node for a given action
  std::unique_ptr<Node> create_child_node(const Node& parent,
                                          const Action& action) const;

  // Create a child node for a sampled chance outcome
  std::unique_ptr<Node> create_chance_child_node(
      const Node& parent,
      const std::vector<Card>& cards) const;
  
  // Get legal actions at a given state
  std::vector<Action> get_legal_actions(const BoardState& state) const;
  
  // Apply an action to a state to get the next state
  BoardState apply_action(const BoardState& state, const Action& action) const;
  
  // Get the utility at a terminal state
  double get_utility(const BoardState& state, const Hand& player_a_hand,
                     const Hand& player_b_hand) const;
  
  // Check if a state is terminal
  bool is_terminal(const BoardState& state) const;
  
  // Get the player to act at a given state
  int get_player_to_act(const BoardState& state) const;
  
  // Check if a betting round is over
  bool is_betting_round_over(const BoardState& state) const;
  
  // Check if the hand is over
  bool is_hand_over(const BoardState& state) const;

private:
  std::unique_ptr<Node> root_;
  PokerConfig config_;
  HandEvaluator hand_evaluator_;
};

} // namespace poker
