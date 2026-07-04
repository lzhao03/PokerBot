#include "src/game_tree.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectInvalidAction(GameTree& tree,
                         const BoardState& state,
                         const Action& action,
                         const char* message) {
  bool threw = false;
  try {
    tree.apply_action(state, action);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, message);
}

void ExpectInvalidActionKey(const Action& action, const char* message) {
  bool threw = false;
  try {
    (void)GameTree::action_key(action);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, message);
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

void AddCard(BoardState& state, int rank, Suit suit) {
  *state.add_cards() = MakeCard(rank, suit);
}

Hand MakeHand(int first_rank, Suit first_suit, int second_rank, Suit second_suit) {
  Hand hand;
  *hand.add_cards() = MakeCard(first_rank, first_suit);
  *hand.add_cards() = MakeCard(second_rank, second_suit);
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
  AddCard(state, 2, Suit::HEARTS);
  AddCard(state, 7, Suit::DIAMONDS);
  AddCard(state, 9, Suit::CLUBS);
  AddCard(state, 11, Suit::SPADES);
  AddCard(state, 12, Suit::DIAMONDS);
  return state;
}

void CheckCallAndCheck(GameTree& tree) {
  BoardState called = tree.apply_action(PreflopState(), MakeAction(ActionType::CALL));

  Expect(called.stack_a() == 98, "call subtracts chips from caller");
  Expect(called.stack_b() == 98, "call leaves opponent stack alone");
  Expect(called.pot() == 4, "call adds outstanding chips to pot");
  Expect(called.player_contribution(0) == 2, "call matches contribution");
  Expect(called.player_to_act() == 1, "call passes action");
  Expect(called.history().actions_size() == 1, "call is recorded");
  Expect(called.history().actions(0).amount() == 1, "call records committed chips");
  Expect(called.history().actions(0).player() == 0, "call records acting player");
  Expect(!tree.is_betting_round_over(called), "call alone does not close preflop");

  BoardState checked = tree.apply_action(called, MakeAction(ActionType::CHECK));
  Expect(checked.pot() == 4, "check leaves pot alone");
  Expect(checked.player_to_act() == 0, "check passes action");
  Expect(checked.history().actions_size() == 2, "check is recorded");
  Expect(tree.is_betting_round_over(checked), "call/check closes preflop");
}

void CheckBetAndCall(GameTree& tree) {
  BoardState bet = tree.apply_action(FlopState(), MakeAction(ActionType::BET, 5));

  Expect(bet.stack_b() == 93, "bet subtracts chips from bettor");
  Expect(bet.pot() == 9, "bet adds chips to pot");
  Expect(bet.player_contribution(1) == 7, "bet adds to contribution");
  Expect(bet.player_to_act() == 0, "bet passes action");

  BoardState called = tree.apply_action(bet, MakeAction(ActionType::CALL));
  Expect(called.stack_a() == 93, "call after bet subtracts matching chips");
  Expect(called.pot() == 14, "call after bet adds matching chips to pot");
  Expect(called.player_contribution(0) == 7, "call after bet matches contribution");
  Expect(called.player_to_act() == 1, "call after bet returns action to first postflop player");
  Expect(tree.is_betting_round_over(called), "bet/call closes postflop");
}

void CheckCheckBetCall(GameTree& tree) {
  BoardState checked = tree.apply_action(FlopState(), MakeAction(ActionType::CHECK));
  BoardState bet = tree.apply_action(checked, MakeAction(ActionType::BET, 5));
  BoardState called = tree.apply_action(bet, MakeAction(ActionType::CALL));

  Expect(called.player_to_act() == 0, "call after checked bet returns action");
  Expect(tree.is_betting_round_over(called), "check/bet/call closes postflop");
}

void CheckRaiseAndFold(GameTree& tree) {
  BoardState raised = tree.apply_action(PreflopState(), MakeAction(ActionType::RAISE, 4));

  Expect(raised.stack_a() == 95, "raise subtracts committed chips");
  Expect(raised.pot() == 7, "raise adds committed chips to pot");
  Expect(raised.player_contribution(0) == 5, "raise increases contribution");
  Expect(raised.player_to_act() == 1, "raise passes action");

  BoardState folded = tree.apply_action(raised, MakeAction(ActionType::FOLD));
  Expect(folded.folded_player() == 1, "fold records folded player");
  Expect(folded.player_to_act() == -1, "fold clears player to act");
  Expect(folded.history().actions_size() == 2, "fold is recorded");
  Expect(tree.is_terminal(folded), "fold is terminal");
  Expect(tree.get_utility(folded, Hand(), Hand()) == 2,
         "fold utility is net chips for player A");
}

void CheckLegalActions(GameTree& tree) {
  std::vector<Action> actions = tree.get_legal_actions(PreflopState());
  Expect(actions.size() == 4, "preflop facing blind has fold/call/raise/all-in");
  Expect(actions[0].action() == ActionType::FOLD, "first legal action is fold");
  Expect(actions[1].action() == ActionType::CALL, "second legal action is call");
  Expect(actions[1].amount() == 1, "call amount is outstanding blind");
  Expect(actions[2].action() == ActionType::RAISE, "third legal action is raise");
  Expect(actions[3].action() == ActionType::ALL_IN, "fourth legal action is all-in");
  Expect(actions[3].amount() == 99, "all-in amount is the remaining stack");
}

void CheckLegalActionsApplyAndConserveChips(GameTree& tree) {
  std::vector<BoardState> states = {PreflopState(), FlopState()};
  for (const BoardState& state : states) {
    int total_chips = state.stack_a() + state.stack_b() + state.pot();
    std::vector<Action> actions = tree.get_legal_actions(state);
    Expect(!actions.empty(), "test states should have legal actions");
    for (const Action& action : actions) {
      BoardState next = tree.apply_action(state, action);
      Expect(next.history().actions_size() ==
                 state.history().actions_size() + 1,
             "legal action should append to history");
      const Action& applied =
          next.history().actions(next.history().actions_size() - 1);
      Expect(applied.action() == action.action(),
             "applied legal action should keep action type");
      Expect(applied.player() == state.player_to_act(),
             "applied legal action should record acting player");
      Expect(next.stack_a() + next.stack_b() + next.pot() == total_chips,
             "legal action should conserve chips");
    }
  }
}

void CheckAllInAction(GameTree& tree) {
  BoardState all_in = tree.apply_action(PreflopState(), MakeAction(ActionType::ALL_IN));

  Expect(all_in.stack_a() == 0, "all-in commits the remaining stack");
  Expect(all_in.pot() == 102, "all-in adds remaining chips to the pot");
  Expect(all_in.player_contribution(0) == 100,
         "all-in updates total player contribution");
  Expect(all_in.all_in(), "all-in marks the state all-in");
  Expect(all_in.player_to_act() == 1, "all-in passes action");
  Expect(all_in.history().actions(0).action() == ActionType::ALL_IN,
         "all-in is recorded");
  Expect(all_in.history().actions(0).amount() == 99,
         "all-in records committed chips");
}

void CheckShortAllInCallClosesRound(GameTree& tree) {
  BoardState state = FlopState();
  state.set_stack_a(5);

  BoardState bet = tree.apply_action(state, MakeAction(ActionType::BET, 10));
  BoardState called = tree.apply_action(bet, MakeAction(ActionType::CALL));

  Expect(called.stack_a() == 0, "short all-in call commits caller stack");
  Expect(called.player_contribution(0) == 7,
         "short all-in call records partial contribution");
  Expect(called.player_contribution(1) == 12,
         "bettor keeps larger contribution");
  Expect(called.all_in(), "short call marks state all-in");
  Expect(tree.is_betting_round_over(called),
         "short all-in call closes heads-up betting");
  Expect(tree.get_player_to_act(called) == -1,
         "short all-in call leaves no player action");

  GameTree::Node& root = tree.build_tree(called);
  Expect(root.is_chance_node, "short all-in call can run out the board");
}

void CheckAllInEquivalentActionsAreDeduped() {
  PokerConfig open_config;
  open_config.add_bet_sizes(24.5);
  GameTree open_tree(open_config);

  std::vector<Action> open_actions = open_tree.get_legal_actions(FlopState());
  Expect(open_actions.size() == 2, "full-stack bet size should become all-in only");
  Expect(open_actions[0].action() == ActionType::CHECK, "first open action is check");
  Expect(open_actions[1].action() == ActionType::ALL_IN, "second open action is all-in");

  PokerConfig raise_config;
  raise_config.add_bet_sizes(32.7);
  GameTree raise_tree(raise_config);

  std::vector<Action> raise_actions = raise_tree.get_legal_actions(PreflopState());
  Expect(raise_actions.size() == 3, "full-stack raise size should become all-in only");
  Expect(raise_actions[0].action() == ActionType::FOLD, "first facing action is fold");
  Expect(raise_actions[1].action() == ActionType::CALL, "second facing action is call");
  Expect(raise_actions[2].action() == ActionType::ALL_IN, "third facing action is all-in");
}

void CheckFullStackBetRaiseUseAllIn(GameTree& tree) {
  ExpectInvalidAction(tree, FlopState(), MakeAction(ActionType::BET, 98),
                      "full-stack bet should use all-in action");
  ExpectInvalidAction(tree, PreflopState(), MakeAction(ActionType::RAISE, 99),
                      "full-stack raise should use all-in action");
}

void CheckDuplicateConcreteBetSizesAreDeduped() {
  PokerConfig config;
  config.add_bet_sizes(0.5);
  config.add_bet_sizes(0.51);
  GameTree tree(config);

  std::vector<Action> open_actions = tree.get_legal_actions(FlopState());
  Expect(open_actions.size() == 3, "duplicate concrete bet sizes should collapse");
  Expect(open_actions[1].action() == ActionType::BET, "open action includes bet");
  Expect(open_actions[1].amount() == 2, "duplicate bet amount is kept once");

  std::vector<Action> raise_actions = tree.get_legal_actions(PreflopState());
  Expect(raise_actions.size() == 4, "duplicate concrete raise sizes should collapse");
  Expect(raise_actions[2].action() == ActionType::RAISE, "facing action includes raise");
  Expect(raise_actions[2].amount() == 2, "duplicate raise amount is kept once");
}

void CheckStreetBetSizesOverrideGlobal() {
  PokerConfig config;
  config.add_bet_sizes(0.5);
  config.add_flop_bet_sizes(1.0);
  GameTree tree(config);

  std::vector<Action> preflop_actions = tree.get_legal_actions(PreflopState());
  Expect(preflop_actions[2].action() == ActionType::RAISE,
         "global size should still apply preflop");
  Expect(preflop_actions[2].amount() == 2,
         "preflop raise should use the global size");

  std::vector<Action> flop_actions = tree.get_legal_actions(FlopState());
  Expect(flop_actions[1].action() == ActionType::BET,
         "flop override should add a bet");
  Expect(flop_actions[1].amount() == 4,
         "flop bet should use the street-specific size");
}

void CheckRaisesRemainAvailableAfterPriorRaise() {
  PokerConfig config;
  config.add_bet_sizes(0.5);
  GameTree tree(config);

  BoardState raised =
      tree.apply_action(PreflopState(), MakeAction(ActionType::RAISE, 4));
  std::vector<Action> actions = tree.get_legal_actions(raised);

  Expect(actions.size() == 4, "facing action should include regular raise");
  Expect(actions[0].action() == ActionType::FOLD, "facing action keeps fold");
  Expect(actions[1].action() == ActionType::CALL, "facing action keeps call");
  Expect(actions[2].action() == ActionType::RAISE,
         "prior raise should not hide regular raise");
  Expect(actions[2].amount() == 6,
         "regular raise amount should use configured pot fraction");
  Expect(actions[3].action() == ActionType::ALL_IN,
         "facing action keeps all-in");
}

void CheckActionKeyRejectsCollidingAmounts() {
  ExpectInvalidActionKey(MakeAction(ActionType::CALL, -1),
                         "action key should reject negative amounts");
  ExpectInvalidActionKey(MakeAction(ActionType::CALL, 1000000),
                         "action key should reject million-chip amounts");
}

void CheckShowdownUtility(GameTree& tree) {
  BoardState showdown = ShowdownState();
  Hand player_a = MakeHand(14, Suit::HEARTS, 14, Suit::SPADES);
  Hand player_b = MakeHand(13, Suit::HEARTS, 13, Suit::SPADES);

  Expect(tree.is_terminal(showdown), "closed river action is terminal");
  Expect(tree.get_utility(showdown, player_a, player_b) == 10,
         "showdown utility is net chips for player A");
}

void CheckChanceAdvancesStreet(GameTree& tree) {
  BoardState closed_preflop =
      tree.apply_action(PreflopState(), MakeAction(ActionType::CALL));
  closed_preflop = tree.apply_action(closed_preflop, MakeAction(ActionType::CHECK));

  GameTree::Node& root = tree.build_tree(closed_preflop);
  Expect(root.is_chance_node, "closed preflop action creates chance node");

  std::vector<Card> flop = {
      MakeCard(8, Suit::HEARTS),
      MakeCard(9, Suit::CLUBS),
      MakeCard(10, Suit::SPADES),
  };
  GameTree::Node& child_ref =
      tree.create_chance_child_node(root, 0, flop);

  Expect(child_ref.state.street() == Street::FLOP, "chance advances preflop to flop");
  Expect(child_ref.state.cards_size() == 3, "chance adds sampled flop cards");
  Expect(child_ref.state.cards(0).rank() == 8, "chance keeps sampled card order");
  Expect(child_ref.state.history().actions_size() == 0, "chance clears street action history");
  Expect(child_ref.player_to_act == 1, "flop action starts with player 1");
  Expect(!child_ref.legal_actions.empty(), "chance child has legal player actions");
}

}  // namespace
}  // namespace poker

int main() {
  poker::PokerConfig config;
  config.add_bet_sizes(0.5);

  poker::GameTree tree(config);
  poker::CheckCallAndCheck(tree);
  poker::CheckBetAndCall(tree);
  poker::CheckCheckBetCall(tree);
  poker::CheckRaiseAndFold(tree);
  poker::CheckLegalActions(tree);
  poker::CheckLegalActionsApplyAndConserveChips(tree);
  poker::CheckAllInAction(tree);
  poker::CheckShortAllInCallClosesRound(tree);
  poker::CheckAllInEquivalentActionsAreDeduped();
  poker::CheckFullStackBetRaiseUseAllIn(tree);
  poker::CheckDuplicateConcreteBetSizesAreDeduped();
  poker::CheckStreetBetSizesOverrideGlobal();
  poker::CheckRaisesRemainAvailableAfterPriorRaise();
  poker::CheckActionKeyRejectsCollidingAmounts();
  poker::CheckShowdownUtility(tree);
  poker::CheckChanceAdvancesStreet(tree);
  return 0;
}
