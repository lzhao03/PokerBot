#include "src/game_tree.h"
#include "src/hand_evaluator.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace poker {

namespace {

constexpr int kPlayerCount = 2;
constexpr int kActionKeyMultiplier = 1000000;

bool IsPlayer(int player) {
  return player == 0 || player == 1;
}

int Opponent(int player) {
  return 1 - player;
}

int GetStack(const BoardState& state, int player) {
  return player == 0 ? state.stack_a() : state.stack_b();
}

void SetStack(BoardState& state, int player, int stack) {
  if (player == 0) {
    state.set_stack_a(stack);
  } else {
    state.set_stack_b(stack);
  }
}

void EnsureHeadsUpContributions(BoardState& state) {
  while (state.player_contribution_size() < kPlayerCount) {
    state.add_player_contribution(0.0);
  }
}

double Contribution(const BoardState& state, int player) {
  return state.player_contribution_size() > player
      ? state.player_contribution(player)
      : 0.0;
}

int OutstandingToCall(const BoardState& state, int player) {
  return std::max(
      0, static_cast<int>(Contribution(state, Opponent(player)) -
                          Contribution(state, player)));
}

int FirstPlayerForStreet(const BoardState& state) {
  return state.street() == Street::PREFLOP ? 0 : 1;
}

bool BoardComplete(const BoardState& state) {
  return state.street() == Street::RIVER && state.cards_size() >= 5;
}

int ConcreteBetAmount(const BoardState& state, double size) {
  if (size <= 0.0) {
    return 0;
  }
  return std::max(1, static_cast<int>(std::max(1, state.pot()) * size));
}

int StreetBetSizesSize(const PokerConfig& config, Street street) {
  switch (street) {
    case Street::PREFLOP:
      return config.preflop_bet_sizes_size();
    case Street::FLOP:
      return config.flop_bet_sizes_size();
    case Street::TURN:
      return config.turn_bet_sizes_size();
    case Street::RIVER:
      return config.river_bet_sizes_size();
    default:
      return 0;
  }
}

double BetSizeForStreet(const PokerConfig& config, Street street, int index) {
  if (StreetBetSizesSize(config, street) == 0) {
    return config.bet_sizes(index);
  }

  switch (street) {
    case Street::PREFLOP:
      return config.preflop_bet_sizes(index);
    case Street::FLOP:
      return config.flop_bet_sizes(index);
    case Street::TURN:
      return config.turn_bet_sizes(index);
    case Street::RIVER:
      return config.river_bet_sizes(index);
    default:
      return config.bet_sizes(index);
  }
}

int BetSizesSize(const PokerConfig& config, Street street) {
  int street_sizes = StreetBetSizesSize(config, street);
  return street_sizes > 0 ? street_sizes : config.bet_sizes_size();
}

int RequestedChips(const Action& action) {
  return static_cast<int>(action.amount());
}

void AddActionIfMissing(std::vector<Action>& actions, ActionType type, int amount) {
  bool exists =
      std::any_of(actions.begin(), actions.end(), [&](const Action& action) {
        return action.action() == type &&
               static_cast<int>(action.amount()) == amount;
      });
  if (exists) {
    return;
  }

  Action new_action;
  new_action.set_action(type);
  new_action.set_amount(amount);
  actions.push_back(new_action);
}

void SetLegalActions(GameTree::Node& node, std::vector<Action> actions) {
  node.legal_actions = std::move(actions);
  node.legal_action_ids.clear();
  node.legal_action_ids.reserve(node.legal_actions.size());
  for (const Action& action : node.legal_actions) {
    node.legal_action_ids.push_back(GameTree::action_key(action));
  }
}

int CommitChips(BoardState& state, int player, int requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action amount must be positive");
  }

  int committed = std::min(requested, GetStack(state, player));
  (*state.mutable_player_contribution())[player] += committed;
  SetStack(state, player, GetStack(state, player) - committed);
  state.set_pot(state.pot() + committed);
  if (GetStack(state, player) == 0) {
    state.set_all_in(true);
  }
  return committed;
}

void AdvanceStreet(BoardState& state, const std::vector<Card>& cards) {
  switch (state.street()) {
    case Street::PREFLOP:
      state.set_street(Street::FLOP);
      break;
    case Street::FLOP:
      state.set_street(Street::TURN);
      break;
    case Street::TURN:
      state.set_street(Street::RIVER);
      break;
    case Street::RIVER:
      break;
    default:
      break;
  }

  for (const Card& card : cards) {
    *state.add_cards() = card;
  }
  state.mutable_history()->mutable_actions()->Clear();
  state.set_player_to_act(FirstPlayerForStreet(state));
}

}  // namespace

GameTree::GameTree(const PokerConfig& config)
  : config_(config) {
}

int GameTree::action_key(const Action& action) {
  return static_cast<int>(action.action()) * kActionKeyMultiplier +
         static_cast<int>(std::lround(action.amount()));
}

GameTree::Node& GameTree::root() {
  if (!root_id_.has_value()) {
    throw std::logic_error("Game tree root has not been built");
  }
  return node(*root_id_);
}

const GameTree::Node& GameTree::root() const {
  if (!root_id_.has_value()) {
    throw std::logic_error("Game tree root has not been built");
  }
  return node(*root_id_);
}

GameTree::Node& GameTree::build_tree(const BoardState& initial_state) {
  node_blocks_.clear();
  node_count_ = 0;
  root_id_ = node_count_;
  Node& root = add_node(Node());
  root.state = initial_state;
  root.is_terminal = is_terminal(initial_state);
  root.player_to_act = get_player_to_act(initial_state);
  
  if (root.is_terminal) {
    root.utility = 0.0;
  } else if (root.player_to_act == -1) {
    root.is_chance_node = true;
  } else {
    SetLegalActions(root, get_legal_actions(initial_state));
  }
  
  return root;
}

std::vector<Action> GameTree::get_legal_actions(const BoardState& state) const {
  std::vector<Action> actions;
  
  // If the hand is over, no actions are legal
  if (is_terminal(state)) {
    return actions;
  }
  
  // Get the player to act
  int player = get_player_to_act(state);
  
  // If it's a chance node, return empty list (chance actions are handled separately)
  if (!IsPlayer(player)) {
    return actions;
  }
  int stack = GetStack(state, player);
  if (stack <= 0) {
    return actions;
  }
  
  int to_call = OutstandingToCall(state, player);
  if (to_call > 0) {
    Action fold_action;
    fold_action.set_action(ActionType::FOLD);
    fold_action.set_amount(0);
    actions.push_back(fold_action);

    Action call_action;
    call_action.set_action(ActionType::CALL);
    call_action.set_amount(std::min(to_call, stack));
    actions.push_back(call_action);

    for (int i = 0; i < BetSizesSize(config_, state.street()); ++i) {
      int raise_amount =
          to_call + ConcreteBetAmount(
                        state, BetSizeForStreet(config_, state.street(), i));
      if (raise_amount < stack) {
        AddActionIfMissing(actions, ActionType::RAISE, raise_amount);
      }
    }
    if (stack > to_call) {
      AddActionIfMissing(actions, ActionType::ALL_IN, stack);
    }
  } else {
    Action check_action;
    check_action.set_action(ActionType::CHECK);
    check_action.set_amount(0);
    actions.push_back(check_action);

    for (int i = 0; i < BetSizesSize(config_, state.street()); ++i) {
      int bet_amount =
          ConcreteBetAmount(state, BetSizeForStreet(config_, state.street(), i));
      if (bet_amount < stack) {
        AddActionIfMissing(actions, ActionType::BET, bet_amount);
      }
    }
    AddActionIfMissing(actions, ActionType::ALL_IN, stack);
  }
  
  return actions;
}

BoardState GameTree::apply_action(const BoardState& state, const Action& action) const {
  BoardState new_state = state;
  EnsureHeadsUpContributions(new_state);

  int player = new_state.player_to_act();
  if (!IsPlayer(player)) {
    player = get_player_to_act(new_state);
  }
  if (!IsPlayer(player)) {
    throw std::invalid_argument("No player can act in this state");
  }
  if (new_state.folded_player() >= 0) {
    throw std::invalid_argument("Cannot act after a player has folded");
  }
  if (GetStack(new_state, player) <= 0) {
    throw std::invalid_argument("Player has no chips to act");
  }

  int opponent = Opponent(player);
  int to_call = OutstandingToCall(new_state, player);
  Action applied = action;
  applied.set_player(player);
  
  // Handle different action types
  switch (action.action()) {
    case ActionType::FOLD: {
      applied.set_amount(0);
      new_state.set_folded_player(player);
      new_state.set_player_to_act(-1);
      break;
    }
    case ActionType::CHECK: {
      if (to_call != 0) {
        throw std::invalid_argument("Cannot check facing a bet");
      }
      applied.set_amount(0);
      new_state.set_player_to_act(opponent);
      break;
    }
    case ActionType::CALL: {
      if (to_call == 0) {
        throw std::invalid_argument("Cannot call without a bet");
      }
      int committed = CommitChips(new_state, player, to_call);
      applied.set_amount(committed);
      new_state.set_player_to_act(opponent);
      break;
    }
    case ActionType::BET: {
      if (to_call != 0) {
        throw std::invalid_argument("Cannot bet facing a bet");
      }
      int requested = RequestedChips(action);
      if (requested >= GetStack(new_state, player)) {
        throw std::invalid_argument("Use all-in for full-stack bets");
      }
      int committed = CommitChips(new_state, player, requested);
      applied.set_amount(committed);
      new_state.set_player_to_act(opponent);
      break;
    }
    case ActionType::RAISE: {
      if (to_call == 0) {
        throw std::invalid_argument("Cannot raise without a bet");
      }
      int requested = RequestedChips(action);
      if (requested <= to_call || GetStack(new_state, player) <= to_call) {
        throw std::invalid_argument("Raise must exceed the call amount");
      }
      if (requested >= GetStack(new_state, player)) {
        throw std::invalid_argument("Use all-in for full-stack raises");
      }
      int committed = CommitChips(new_state, player, requested);
      applied.set_amount(committed);
      new_state.set_player_to_act(opponent);
      break;
    }
    case ActionType::ALL_IN: {
      int committed = CommitChips(new_state, player, GetStack(new_state, player));
      applied.set_amount(committed);
      new_state.set_player_to_act(opponent);
      break;
    }
    default:
      throw std::invalid_argument("Unknown action type");
  }

  *new_state.mutable_history()->add_actions() = applied;
  
  return new_state;
}

double GameTree::get_utility(const BoardState& state,
                             const Hand& player_a_hand,
                             const Hand& player_b_hand) const {
  double player_a_contribution = Contribution(state, 0);

  // If a player has folded, return the appropriate utility
  if (state.folded_player() >= 0) {
    if (state.folded_player() == 0) {
      // Player A folded, they lose their contribution to the pot
      return -player_a_contribution;
    } else {
      // Player B folded, Player A wins the pot net of chips already committed
      return state.pot() - player_a_contribution;
    }
  }
  
  // If we've reached showdown, compare the hands
  // Check if we have enough cards for a valid hand evaluation
  int total_cards = player_a_hand.cards_size() + state.cards_size();
  if (total_cards < 5) {
    // Not enough cards for a valid hand, return a default utility
    // In a real implementation, this would be based on the current pot and player contributions
    return 0.0;
  }
  
  try {
    int comparison = hand_evaluator_.compare_hands(player_a_hand, player_b_hand, state);
    
    if (comparison > 0) {
      // Player A wins
      return state.pot() - player_a_contribution;
    } else if (comparison < 0) {
      // Player B wins
      return -player_a_contribution;
    } else {
      // Tie, split the pot
      return (state.pot() / 2.0) - player_a_contribution;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error evaluating hands: " << e.what() << std::endl;
    return 0.0;
  }
}

bool GameTree::is_terminal(const BoardState& state) const {
  // A state is terminal if:
  // 1. A player has folded, or
  // 2. The hand is over (all cards dealt and all betting rounds complete)
  return state.folded_player() >= 0 || is_hand_over(state);
}

int GameTree::get_player_to_act(const BoardState& state) const {
  if (is_terminal(state)) {
    return -1;
  }
  if (is_betting_round_over(state)) {
    return -1;
  }

  if (IsPlayer(state.player_to_act())) {
    return state.player_to_act();
  }
  
  return FirstPlayerForStreet(state);
}

GameTree::Node& GameTree::create_child_node(Node& parent,
                                            int child_key,
                                            const Action& action) {
  return add_child(parent, child_key, make_child_node(parent, action));
}

GameTree::Node GameTree::make_child_node(const Node& parent,
                                         const Action& action) const {
  Node child;
  
  child.state = apply_action(parent.state, action);
  child.player_to_act = get_player_to_act(child.state);
  child.is_terminal = is_terminal(child.state);
  child.is_chance_node = !child.is_terminal && child.player_to_act == -1;
  
  // If the node is terminal, compute the utility
  if (child.is_terminal) {
    // This will be computed when needed in get_utility
    child.utility = 0.0;
  } else if (!child.is_chance_node) {
    SetLegalActions(child, get_legal_actions(child.state));
  }
  
  return child;
}

GameTree::Node& GameTree::create_chance_child_node(
    Node& parent,
    int child_key,
    const std::vector<Card>& cards) {
  return add_child(parent, child_key, make_chance_child_node(parent, cards));
}

GameTree::Node GameTree::make_chance_child_node(
    const Node& parent,
    const std::vector<Card>& cards) const {
  if (!parent.is_chance_node) {
    throw std::invalid_argument("Parent node is not a chance node");
  }

  Node child;
  child.state = parent.state;
  AdvanceStreet(child.state, cards);
  child.is_terminal = is_terminal(child.state);
  child.player_to_act = get_player_to_act(child.state);
  child.is_chance_node = !child.is_terminal && child.player_to_act == -1;

  if (child.is_terminal) {
    child.utility = 0.0;
  } else if (!child.is_chance_node) {
    SetLegalActions(child, get_legal_actions(child.state));
  }

  return child;
}

GameTree::Node& GameTree::add_child(Node& parent, int child_key, Node child) {
  auto existing = parent.children.find(child_key);
  if (existing != parent.children.end()) {
    return node(existing->second);
  }

  NodeId child_id = node_count_;
  Node& child_ref = add_node(std::move(child));
  parent.children.emplace(child_key, child_id);
  return child_ref;
}

GameTree::Node& GameTree::node(NodeId id) {
  if (id >= node_count_) {
    throw std::logic_error("Invalid game tree node ID " +
                           std::to_string(id) + " for arena size " +
                           std::to_string(node_count_));
  }
  return node_blocks_[id / kNodeBlockSize]->nodes[id % kNodeBlockSize];
}

const GameTree::Node& GameTree::node(NodeId id) const {
  if (id >= node_count_) {
    throw std::logic_error("Invalid game tree node ID " +
                           std::to_string(id) + " for arena size " +
                           std::to_string(node_count_));
  }
  return node_blocks_[id / kNodeBlockSize]->nodes[id % kNodeBlockSize];
}

GameTree::Node& GameTree::add_node(Node node) {
  if (node_blocks_.empty() ||
      node_blocks_.back()->nodes.size() == kNodeBlockSize) {
    node_blocks_.push_back(std::make_unique<NodeBlock>());
  }
  NodeBlock& block = *node_blocks_.back();
  block.nodes.push_back(std::move(node));
  ++node_count_;
  return block.nodes.back();
}

bool GameTree::is_betting_round_over(const BoardState& state) const {
  if (state.folded_player() >= 0) {
    return true;
  }
  bool calls_matched =
      OutstandingToCall(state, 0) == 0 && OutstandingToCall(state, 1) == 0;
  if (state.all_in()) {
    int player = state.player_to_act();
    return calls_matched || !IsPlayer(player) || GetStack(state, player) == 0 ||
           OutstandingToCall(state, player) == 0;
  }
  if (state.history().actions_size() == 0 || !calls_matched) {
    return false;
  }
  
  const Action& last = state.history().actions(state.history().actions_size() - 1);
  if (last.action() == ActionType::CALL) {
    return state.history().actions_size() > 1;
  }
  return last.action() == ActionType::CHECK &&
         state.player_to_act() == FirstPlayerForStreet(state);
}

bool GameTree::is_hand_over(const BoardState& state) const {
  // A hand is over when all cards are dealt and all betting rounds are complete
  return BoardComplete(state) && is_betting_round_over(state);
}

} // namespace poker
