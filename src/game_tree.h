#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "src/hand_evaluator.h"
#include "src/poker_types.h"

namespace poker {

class GameTree {
public:
  using NodeId = std::size_t;

  // Maximum number of legal actions at any decision node.
  // With 3 bet sizes: fold/call + 3 raises + all-in = 6 (facing bet),
  // or check + 3 bets + all-in = 5 (no bet). 8 gives headroom.
  static constexpr int kMaxActionsPerNode = 8;
  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();

  // Per-action entry stored inline in each Node.
  struct ActionEntry {
    GameAction action;
    int         key      = 0;
    NodeId      child_id = kInvalidNodeId; // kInvalidNodeId = not yet created
  };

  // Node in the game tree representing a decision point.
  // All action/child data is stored inline — no heap allocation.
  struct Node {
    static constexpr uint32_t kInvalidPublicStateId =
        std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t kInvalidBettingHistoryId =
        std::numeric_limits<uint32_t>::max();

    GameState state;
    bool is_terminal   = false;
    bool is_chance_node = false;
    int  player_to_act = -1; // 0 = player A, 1 = player B, -1 = chance

    // Inline action table (bounded by kMaxActionsPerNode).
    uint8_t action_count = 0;
    std::array<ActionEntry, kMaxActionsPerNode> actions = {};

    uint32_t betting_history_id = kInvalidBettingHistoryId;
    uint32_t public_state_id = kInvalidPublicStateId;
    NodeId   id              = 0;

    // For terminal nodes only.
    double utility = 0.0;

    // --- helpers that mirror the old vector/map interface ---

    // Append a legal action (no child yet). Throws if kMaxActionsPerNode exceeded.
    void add_action(const GameAction& action, int key);

    // Find the child_id for a given key, or kInvalidNodeId if not found.
    NodeId find_child(int key) const;

    // Set the child_id for a given key. Key must already be present.
    void set_child(int key, NodeId child_id);

    NodeId child_for_action_index(int action_index) const;
    void set_child_for_action_index(int action_index, NodeId child_id);

    // Iterate over legal actions as a span-like view (used by callers that
    // previously iterated node.legal_actions).
    const ActionEntry* actions_begin() const { return actions.data(); }
    const ActionEntry* actions_end()   const { return actions.data() + action_count; }
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

  Node& create_child_node(Node& parent, int action_index);

  // Create a child node for a sampled chance outcome
  Node& create_chance_child_node(
      Node& parent,
      int child_key,
      absl::Span<const CardId> cards);

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

  // Total number of nodes currently allocated in the tree arena.
  size_t node_count() const { return node_count_; }

  // Look up a chance-node child by parent ID and chance key.
  // Returns kInvalidNodeId if not found.
  NodeId find_chance_child(NodeId parent_id, int child_key) const;

private:
  static constexpr std::size_t kNodeBlockSize = 4096;

  struct NodeBlock {
    NodeBlock() { nodes.reserve(kNodeBlockSize); }

    std::vector<Node> nodes;
  };

  Node make_child_node(const Node& parent, const GameAction& action) const;
  Node make_chance_child_node(
      const Node& parent,
      absl::Span<const CardId> cards) const;
  Node& add_node(Node node);

  // Helpers for the chance-children side-table.
  static uint64_t ChanceChildKey(NodeId parent_id, int child_key) {
    return (static_cast<uint64_t>(parent_id) << 32) |
           static_cast<uint64_t>(static_cast<uint32_t>(child_key));
  }
  void set_chance_child(NodeId parent_id, int child_key, NodeId child_id);

  std::vector<std::unique_ptr<NodeBlock>> node_blocks_;
  std::size_t node_count_ = 0;
  std::optional<NodeId> root_id_;
  SolverConfig config_;
  HandEvaluator hand_evaluator_;

  // Lookup table for chance-node children: (parent_id, child_key) -> child_id.
  // Player-action children are stored inline in Node::actions; chance children
  // can be numerous (board runouts) so they live here to avoid per-node heap
  // allocation while still allowing arbitrary counts.
  absl::flat_hash_map<uint64_t, NodeId> chance_children_;
};

} // namespace poker
