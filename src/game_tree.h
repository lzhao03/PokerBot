#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "src/hand_evaluator.h"
#include "src/poker.pb.h"

namespace poker {

class GameTree {
public:
  using NodeId = std::size_t;

  // Node in the game tree representing a decision point
  struct Node {
    BoardState state;
    bool is_terminal;
    bool is_chance_node;
    int player_to_act; // 0 for player A, 1 for player B, -1 for chance
    std::vector<Action> legal_actions;
    std::vector<int> legal_action_ids;
    absl::flat_hash_map<int, NodeId> children; // Action ID -> node arena ID
    NodeId id = 0;
    
    // For terminal nodes
    double utility;
    
    Node() : is_terminal(false), is_chance_node(false), player_to_act(-1), utility(0.0) {}
  };
  
  // Constructor
  GameTree(const PokerConfig& config);

  static int action_key(const Action& action);
  
  // Build the root node from the initial state
  Node& build_tree(const BoardState& initial_state);
  
  // Get the root node of the tree
  bool has_root() const { return root_id_.has_value(); }
  Node& root();
  const Node& root() const;
  
  // Create a child node for a given action
  Node& create_child_node(Node& parent,
                          int child_key,
                          const Action& action);

  // Create a child node for a sampled chance outcome
  Node& create_chance_child_node(
      Node& parent,
      int child_key,
      const std::vector<Card>& cards);

  // Move an already-built node into the arena and link it as a child.
  Node& add_child(Node& parent, int child_key, Node child);
  Node& node(NodeId id);
  const Node& node(NodeId id) const;
  
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
  static constexpr std::size_t kNodeBlockSize = 4096;

  struct NodeBlock {
    NodeBlock() { nodes.reserve(kNodeBlockSize); }

    std::vector<Node> nodes;
  };

  Node make_child_node(const Node& parent, const Action& action) const;
  Node make_chance_child_node(
      const Node& parent,
      const std::vector<Card>& cards) const;
  Node& add_node(Node node);

  std::vector<std::unique_ptr<NodeBlock>> node_blocks_;
  std::size_t node_count_ = 0;
  std::optional<NodeId> root_id_;
  PokerConfig config_;
  HandEvaluator hand_evaluator_;
};

} // namespace poker
