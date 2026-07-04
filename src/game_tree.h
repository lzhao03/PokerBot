#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "src/hand_evaluator.h"
#include "src/poker_types.h"

namespace poker {

class GameTree {
public:
  using NodeId = std::size_t;

  // Node in the game tree representing a decision point
  struct Node {
    GameState state;
    bool is_terminal;
    bool is_chance_node;
    int player_to_act; // 0 for player A, 1 for player B, -1 for chance
    std::vector<GameAction> legal_actions;
    std::vector<int> legal_action_ids;
    absl::flat_hash_map<int, NodeId> children; // Action ID -> node arena ID
    NodeId id = 0;
    
    // For terminal nodes
    double utility;
    
    Node() : is_terminal(false), is_chance_node(false), player_to_act(-1), utility(0.0) {}
  };
  
  // Constructor
  GameTree(const SolverConfig& config);

  static int action_key(const GameAction& action);
  
  // Build the root node from the initial state
  Node& build_tree(const GameState& initial_state);
  
  // Get the root node of the tree
  bool has_root() const { return root_id_.has_value(); }
  Node& root();
  const Node& root() const;
  
  // Create a child node for a given action
  Node& create_child_node(Node& parent,
                          int child_key,
                          const GameAction& action);

  // Create a child node for a sampled chance outcome
  Node& create_chance_child_node(
      Node& parent,
      int child_key,
      const std::vector<CardId>& cards);

  // Move an already-built node into the arena and link it as a child.
  Node& add_child(Node& parent, int child_key, Node child);
  Node& node(NodeId id);
  const Node& node(NodeId id) const;
  
  // Get legal actions at a given state
  std::vector<GameAction> get_legal_actions(const GameState& state) const;
  
  // Apply an action to a state to get the next state
  GameState apply_action(const GameState& state,
                         const GameAction& action) const;
  
  // Get the utility at a terminal state
  double get_utility(const GameState& state, ComboId player_a_hand,
                     ComboId player_b_hand) const;
  
  // Check if a state is terminal
  bool is_terminal(const GameState& state) const;
  
  // Get the player to act at a given state
  int get_player_to_act(const GameState& state) const;
  
  // Check if a betting round is over
  bool is_betting_round_over(const GameState& state) const;
  
  // Check if the hand is over
  bool is_hand_over(const GameState& state) const;

private:
  static constexpr std::size_t kNodeBlockSize = 4096;

  struct NodeBlock {
    NodeBlock() { nodes.reserve(kNodeBlockSize); }

    std::vector<Node> nodes;
  };

  Node make_child_node(const Node& parent, const GameAction& action) const;
  Node make_chance_child_node(
      const Node& parent,
      const std::vector<CardId>& cards) const;
  Node& add_node(Node node);

  std::vector<std::unique_ptr<NodeBlock>> node_blocks_;
  std::size_t node_count_ = 0;
  std::optional<NodeId> root_id_;
  SolverConfig config_;
  HandEvaluator hand_evaluator_;
};

} // namespace poker
