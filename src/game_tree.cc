#include "src/game_tree.h"
#include "src/hand_evaluator.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace poker {

namespace {

constexpr int kPlayerCount = 2;

bool IsPlayer(int player) {
  return player == 0 || player == 1;
}

int Opponent(int player) {
  return 1 - player;
}

int GetStack(const BoardState& state, int player) {
  return player == 0 ? state.stack_a() : state.stack_b();
}

void SetStack(BoardState* state, int player, int stack) {
  if (player == 0) {
    state->set_stack_a(stack);
  } else {
    state->set_stack_b(stack);
  }
}

void EnsureHeadsUpContributions(BoardState* state) {
  while (state->player_contribution_size() < kPlayerCount) {
    state->add_player_contribution(0.0);
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

int RequestedChips(const Action& action) {
  return static_cast<int>(action.amount());
}

int CommitChips(BoardState* state, int player, int requested) {
  if (requested <= 0) {
    throw std::invalid_argument("Action amount must be positive");
  }

  int committed = std::min(requested, GetStack(*state, player));
  (*state->mutable_player_contribution())[player] += committed;
  SetStack(state, player, GetStack(*state, player) - committed);
  state->set_pot(state->pot() + committed);
  if (GetStack(*state, player) == 0) {
    state->set_all_in(true);
  }
  return committed;
}

void AdvanceStreet(BoardState* state, const std::vector<Card>& cards) {
  switch (state->street()) {
    case Street::PREFLOP:
      state->set_street(Street::FLOP);
      break;
    case Street::FLOP:
      state->set_street(Street::TURN);
      break;
    case Street::TURN:
      state->set_street(Street::RIVER);
      break;
    case Street::RIVER:
      break;
    default:
      break;
  }

  for (const Card& card : cards) {
    *state->add_cards() = card;
  }
  state->mutable_history()->mutable_actions()->Clear();
  state->set_player_to_act(FirstPlayerForStreet(*state));
}

}  // namespace

// Node destructor - recursively delete all children
GameTree::Node::~Node() {
  for (auto& pair : children) {
    delete pair.second;
  }
  children.clear();
}

GameTree::GameTree(const PokerConfig& config) 
  : root_(nullptr), config_(config), hand_evaluator_(new HandEvaluator()) {
}

GameTree::~GameTree() {
  delete hand_evaluator_;
  cleanup();
}

GameTree::Node* GameTree::build_tree(const BoardState& initial_state) {
  // Clean up any existing tree
  cleanup();
  
  // Create the root node
  root_ = new Node();
  root_->state = initial_state;
  root_->is_terminal = is_terminal(initial_state);
  root_->player_to_act = get_player_to_act(initial_state);
  
  if (root_->is_terminal) {
    root_->utility = 0.0;
  } else if (root_->player_to_act == -1) {
    root_->is_chance_node = true;
  } else {
    root_->legal_actions = get_legal_actions(initial_state);
  }
  
  return root_;
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
  if (!IsPlayer(player) || GetStack(state, player) <= 0) {
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
    call_action.set_amount(std::min(to_call, GetStack(state, player)));
    actions.push_back(call_action);

    for (int i = 0; i < config_.bet_sizes_size(); ++i) {
      int raise_amount =
          to_call + ConcreteBetAmount(state, config_.bet_sizes(i));
      if (raise_amount <= GetStack(state, player)) {
        Action raise_action;
        raise_action.set_action(ActionType::RAISE);
        raise_action.set_amount(raise_amount);
        actions.push_back(raise_action);
      }
    }
  } else {
    Action check_action;
    check_action.set_action(ActionType::CHECK);
    check_action.set_amount(0);
    actions.push_back(check_action);

    for (int i = 0; i < config_.bet_sizes_size(); ++i) {
      int bet_amount = ConcreteBetAmount(state, config_.bet_sizes(i));
      if (bet_amount <= GetStack(state, player)) {
        Action bet_action;
        bet_action.set_action(ActionType::BET);
        bet_action.set_amount(bet_amount);
        actions.push_back(bet_action);
      }
    }
  }
  
  return actions;
}

BoardState GameTree::apply_action(const BoardState& state, const Action& action) const {
  BoardState new_state = state;
  EnsureHeadsUpContributions(&new_state);

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
      int committed = CommitChips(&new_state, player, to_call);
      applied.set_amount(committed);
      new_state.set_player_to_act(opponent);
      break;
    }
    case ActionType::BET: {
      if (to_call != 0) {
        throw std::invalid_argument("Cannot bet facing a bet");
      }
      int committed = CommitChips(&new_state, player, RequestedChips(action));
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
      int committed = CommitChips(&new_state, player, requested);
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

double GameTree::get_utility(const BoardState& state, const Hand& player_a_hand, const Hand& player_b_hand) {
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
    int comparison = hand_evaluator_->compare_hands(player_a_hand, player_b_hand, state);
    
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

void GameTree::cleanup() {
  if (root_) {
    delete root_;
    root_ = nullptr;
  }
}

GameTree::Node* GameTree::create_child_node(Node* parent, const Action& action) {
  // Create a new node
  Node* child = new Node();
  
  child->state = apply_action(parent->state, action);
  child->player_to_act = get_player_to_act(child->state);
  child->is_terminal = is_terminal(child->state);
  child->is_chance_node = !child->is_terminal && child->player_to_act == -1;
  
  // If the node is terminal, compute the utility
  if (child->is_terminal) {
    // This will be computed when needed in get_utility
    child->utility = 0.0;
  } else if (!child->is_chance_node) {
    child->legal_actions = get_legal_actions(child->state);
  }
  
  return child;
}

GameTree::Node* GameTree::create_chance_child_node(
    Node* parent, const std::vector<Card>& cards) {
  if (!parent->is_chance_node) {
    throw std::invalid_argument("Parent node is not a chance node");
  }

  Node* child = new Node();
  child->state = parent->state;
  AdvanceStreet(&child->state, cards);
  child->is_terminal = is_terminal(child->state);
  child->player_to_act = get_player_to_act(child->state);
  child->is_chance_node = !child->is_terminal && child->player_to_act == -1;

  if (child->is_terminal) {
    child->utility = 0.0;
  } else if (!child->is_chance_node) {
    child->legal_actions = get_legal_actions(child->state);
  }

  return child;
}

bool GameTree::is_betting_round_over(const BoardState& state) const {
  if (state.folded_player() >= 0) {
    return true;
  }
  bool calls_matched =
      OutstandingToCall(state, 0) == 0 && OutstandingToCall(state, 1) == 0;
  if (state.all_in()) {
    int player = state.player_to_act();
    return calls_matched || !IsPlayer(player) || GetStack(state, player) == 0;
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
