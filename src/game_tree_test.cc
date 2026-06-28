#include "src/game_tree.h"

#include <stdexcept>
#include <vector>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Action MakeAction(ActionType type, int amount = 0) {
  Action action;
  action.set_action(type);
  action.set_amount(amount);
  return action;
}

Card MakeCard(int rank, Suit suit) {
  Card card;
  card.set_rank(rank);
  card.set_suit(suit);
  return card;
}

void AddCard(BoardState* state, int rank, Suit suit) {
  Card* card = state->add_cards();
  card->set_rank(rank);
  card->set_suit(suit);
}

Hand MakeHand(int first_rank, Suit first_suit, int second_rank, Suit second_suit) {
  Hand hand;
  Card* first = hand.add_cards();
  first->set_rank(first_rank);
  first->set_suit(first_suit);
  Card* second = hand.add_cards();
  second->set_rank(second_rank);
  second->set_suit(second_suit);
  return hand;
}

BoardState PreflopState() {
  BoardState state;
  state.set_stack_a(99);
  state.set_stack_b(98);
  state.set_pot(3);
  state.set_street(Street::PREFLOP);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.add_player_contribution(1);
  state.add_player_contribution(2);
  state.set_player_to_act(0);
  return state;
}

BoardState FlopState() {
  BoardState state;
  state.set_stack_a(98);
  state.set_stack_b(98);
  state.set_pot(4);
  state.set_street(Street::FLOP);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.add_player_contribution(2);
  state.add_player_contribution(2);
  state.set_player_to_act(1);
  return state;
}

BoardState ShowdownState() {
  BoardState state;
  state.set_stack_a(90);
  state.set_stack_b(90);
  state.set_pot(20);
  state.set_street(Street::RIVER);
  state.set_all_in(false);
  state.set_folded_player(-1);
  state.add_player_contribution(10);
  state.add_player_contribution(10);
  state.set_player_to_act(1);
  *state.mutable_history()->add_actions() = MakeAction(ActionType::CHECK);
  *state.mutable_history()->add_actions() = MakeAction(ActionType::CHECK);
  AddCard(&state, 2, Suit::HEARTS);
  AddCard(&state, 7, Suit::DIAMONDS);
  AddCard(&state, 9, Suit::CLUBS);
  AddCard(&state, 11, Suit::SPADES);
  AddCard(&state, 12, Suit::DIAMONDS);
  return state;
}

void CheckCallAndCheck(GameTree* tree) {
  BoardState called = tree->apply_action(PreflopState(), MakeAction(ActionType::CALL));

  Expect(called.stack_a() == 98, "call subtracts chips from caller");
  Expect(called.stack_b() == 98, "call leaves opponent stack alone");
  Expect(called.pot() == 4, "call adds outstanding chips to pot");
  Expect(called.player_contribution(0) == 2, "call matches contribution");
  Expect(called.player_to_act() == 1, "call passes action");
  Expect(called.history().actions_size() == 1, "call is recorded");
  Expect(called.history().actions(0).amount() == 1, "call records committed chips");
  Expect(!tree->is_betting_round_over(called), "call alone does not close preflop");

  BoardState checked = tree->apply_action(called, MakeAction(ActionType::CHECK));
  Expect(checked.pot() == 4, "check leaves pot alone");
  Expect(checked.player_to_act() == 0, "check passes action");
  Expect(checked.history().actions_size() == 2, "check is recorded");
  Expect(tree->is_betting_round_over(checked), "call/check closes preflop");
}

void CheckBetAndCall(GameTree* tree) {
  BoardState bet = tree->apply_action(FlopState(), MakeAction(ActionType::BET, 5));

  Expect(bet.stack_b() == 93, "bet subtracts chips from bettor");
  Expect(bet.pot() == 9, "bet adds chips to pot");
  Expect(bet.player_contribution(1) == 7, "bet adds to contribution");
  Expect(bet.player_to_act() == 0, "bet passes action");

  BoardState called = tree->apply_action(bet, MakeAction(ActionType::CALL));
  Expect(called.stack_a() == 93, "call after bet subtracts matching chips");
  Expect(called.pot() == 14, "call after bet adds matching chips to pot");
  Expect(called.player_contribution(0) == 7, "call after bet matches contribution");
  Expect(called.player_to_act() == 1, "call after bet returns action to first postflop player");
  Expect(tree->is_betting_round_over(called), "bet/call closes postflop");
}

void CheckRaiseAndFold(GameTree* tree) {
  BoardState raised = tree->apply_action(PreflopState(), MakeAction(ActionType::RAISE, 4));

  Expect(raised.stack_a() == 95, "raise subtracts committed chips");
  Expect(raised.pot() == 7, "raise adds committed chips to pot");
  Expect(raised.player_contribution(0) == 5, "raise increases contribution");
  Expect(raised.player_to_act() == 1, "raise passes action");

  BoardState folded = tree->apply_action(raised, MakeAction(ActionType::FOLD));
  Expect(folded.folded_player() == 1, "fold records folded player");
  Expect(folded.player_to_act() == -1, "fold clears player to act");
  Expect(folded.history().actions_size() == 2, "fold is recorded");
  Expect(tree->is_terminal(folded), "fold is terminal");
  Expect(tree->get_utility(folded, Hand(), Hand()) == folded.pot(),
         "fold utility pays player A when player B folds");
}

void CheckLegalActions(GameTree* tree) {
  std::vector<Action> actions = tree->get_legal_actions(PreflopState());
  Expect(actions.size() == 3, "preflop facing blind has fold/call/raise");
  Expect(actions[0].action() == ActionType::FOLD, "first legal action is fold");
  Expect(actions[1].action() == ActionType::CALL, "second legal action is call");
  Expect(actions[1].amount() == 1, "call amount is outstanding blind");
  Expect(actions[2].action() == ActionType::RAISE, "third legal action is raise");
}

void CheckShowdownUtility(GameTree* tree) {
  BoardState showdown = ShowdownState();
  Hand player_a = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);

  Expect(tree->is_terminal(showdown), "closed river action is terminal");
  Expect(tree->get_utility(showdown, player_a, player_b) == 20,
         "showdown utility pays pot to winning player A");
}

void CheckChanceAdvancesStreet(GameTree* tree) {
  BoardState closed_preflop =
      tree->apply_action(PreflopState(), MakeAction(ActionType::CALL));
  closed_preflop = tree->apply_action(closed_preflop, MakeAction(ActionType::CHECK));

  GameTree::Node* root = tree->build_tree(closed_preflop);
  Expect(root->is_chance_node, "closed preflop action creates chance node");

  std::vector<Card> flop = {
      MakeCard(8, Suit::HEARTS),
      MakeCard(9, Suit::CLUBS),
      MakeCard(10, Suit::SPADES),
  };
  GameTree::Node* child = tree->create_chance_child_node(root, flop);

  Expect(child->state.street() == Street::FLOP, "chance advances preflop to flop");
  Expect(child->state.cards_size() == 3, "chance adds sampled flop cards");
  Expect(child->state.cards(0).rank() == 8, "chance keeps sampled card order");
  Expect(child->state.history().actions_size() == 0, "chance clears street action history");
  Expect(child->player_to_act == 1, "flop action starts with player 1");
  Expect(!child->legal_actions.empty(), "chance child has legal player actions");

  delete child;
}

}  // namespace
}  // namespace poker

int main() {
  poker::PokerConfig config;
  config.add_bet_sizes(0.5);

  poker::GameTree tree(config);
  poker::CheckCallAndCheck(&tree);
  poker::CheckBetAndCall(&tree);
  poker::CheckRaiseAndFold(&tree);
  poker::CheckLegalActions(&tree);
  poker::CheckShowdownUtility(&tree);
  poker::CheckChanceAdvancesStreet(&tree);
  return 0;
}
