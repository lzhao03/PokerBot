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

GameAction MakeAction(ActionKind type, int amount = 0) {
  return {type, amount, -1};
}

ComboId MakeCombo(int first_rank,
                  SuitKind first_suit,
                  int second_rank,
                  SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

void AddCard(GameState& state, int rank, SuitKind suit) {
  AddBoardCard(state, MakeCardId(rank, suit));
}

SolverConfig TestConfig() {
  SolverConfig config;
  config.bet_sizes.push_back(0.5);
  config.starting_stack_size = 100;
  config.small_blind = 1;
  config.big_blind = 2;
  return config;
}

GameState PreflopState() {
  GameState state;
  state.stack[0] = 99;
  state.stack[1] = 98;
  state.pot = 3;
  state.street = StreetKind::kPreflop;
  state.all_in = false;
  state.folded_player = -1;
  state.player_contribution = {1, 2};
  state.player_to_act = 0;
  return state;
}

GameState FlopState() {
  GameState state;
  state.stack[0] = 98;
  state.stack[1] = 98;
  state.pot = 4;
  state.street = StreetKind::kFlop;
  state.all_in = false;
  state.folded_player = -1;
  state.player_contribution = {2, 2};
  state.player_to_act = 1;
  return state;
}

GameState ShowdownState() {
  GameState state;
  state.stack[0] = 90;
  state.stack[1] = 90;
  state.pot = 20;
  state.street = StreetKind::kRiver;
  state.all_in = false;
  state.folded_player = -1;
  state.player_contribution = {10, 10};
  state.player_to_act = 1;
  state.history.push_back({ActionKind::kCheck, 0, 1});
  state.history.push_back({ActionKind::kCheck, 0, 0});
  AddCard(state, 2, SuitKind::kHearts);
  AddCard(state, 7, SuitKind::kDiamonds);
  AddCard(state, 9, SuitKind::kClubs);
  AddCard(state, 11, SuitKind::kSpades);
  AddCard(state, 12, SuitKind::kDiamonds);
  return state;
}

void ExpectInvalidAction(GameTree& tree,
                         const GameState& state,
                         const GameAction& action,
                         const char* message) {
  bool threw = false;
  try {
    tree.apply_action(state, action);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, message);
}

void ExpectInvalidActionKey(const GameAction& action, const char* message) {
  bool threw = false;
  try {
    (void)GameTree::action_key(action);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Expect(threw, message);
}

void CheckCallAndCheck(GameTree& tree) {
  GameState called =
      tree.apply_action(PreflopState(), MakeAction(ActionKind::kCall));

  Expect(called.stack[0] == 98, "call subtracts chips from caller");
  Expect(called.stack[1] == 98, "call leaves opponent stack alone");
  Expect(called.pot == 4, "call adds outstanding chips to pot");
  Expect(called.player_contribution[0] == 2, "call matches contribution");
  Expect(called.player_to_act == 1, "call passes action");
  Expect(called.history.size() == 1, "call is recorded");
  Expect(called.history[0].amount == 1, "call records committed chips");
  Expect(called.history[0].player == 0, "call records acting player");
  Expect(!tree.is_betting_round_over(called),
         "call alone does not close preflop");

  GameState checked =
      tree.apply_action(called, MakeAction(ActionKind::kCheck));
  Expect(checked.pot == 4, "check leaves pot alone");
  Expect(checked.player_to_act == 0, "check passes action");
  Expect(checked.history.size() == 2, "check is recorded");
  Expect(tree.is_betting_round_over(checked), "call/check closes preflop");
}

void CheckBetAndCall(GameTree& tree) {
  GameState bet =
      tree.apply_action(FlopState(), MakeAction(ActionKind::kBet, 5));

  Expect(bet.stack[1] == 93, "bet subtracts chips from bettor");
  Expect(bet.pot == 9, "bet adds chips to pot");
  Expect(bet.player_contribution[1] == 7, "bet adds to contribution");
  Expect(bet.player_to_act == 0, "bet passes action");

  GameState called = tree.apply_action(bet, MakeAction(ActionKind::kCall));
  Expect(called.stack[0] == 93, "call after bet subtracts matching chips");
  Expect(called.pot == 14, "call after bet adds matching chips to pot");
  Expect(called.player_contribution[0] == 7,
         "call after bet matches contribution");
  Expect(called.player_to_act == 1,
         "call after bet returns action to first postflop player");
  Expect(tree.is_betting_round_over(called), "bet/call closes postflop");
}

void CheckCheckBetCall(GameTree& tree) {
  GameState checked =
      tree.apply_action(FlopState(), MakeAction(ActionKind::kCheck));
  GameState bet = tree.apply_action(checked, MakeAction(ActionKind::kBet, 5));
  GameState called = tree.apply_action(bet, MakeAction(ActionKind::kCall));

  Expect(called.player_to_act == 0, "call after checked bet returns action");
  Expect(tree.is_betting_round_over(called),
         "check/bet/call closes postflop");
}

void CheckRaiseAndFold(GameTree& tree) {
  GameState raised =
      tree.apply_action(PreflopState(), MakeAction(ActionKind::kRaise, 4));

  Expect(raised.stack[0] == 95, "raise subtracts committed chips");
  Expect(raised.pot == 7, "raise adds committed chips to pot");
  Expect(raised.player_contribution[0] == 5,
         "raise increases contribution");
  Expect(raised.player_to_act == 1, "raise passes action");

  GameState folded = tree.apply_action(raised, MakeAction(ActionKind::kFold));
  Expect(folded.folded_player == 1, "fold records folded player");
  Expect(folded.player_to_act == -1, "fold clears player to act");
  Expect(folded.history.size() == 2, "fold is recorded");
  Expect(tree.is_terminal(folded), "fold is terminal");
  Expect(tree.get_utility(folded, 0, 1) == 2,
         "fold utility is net chips for player A");
}

void CheckLegalActions(GameTree& tree) {
  std::vector<GameAction> actions = tree.get_legal_actions(PreflopState());
  Expect(actions.size() == 4, "preflop facing blind has fold/call/raise/all-in");
  Expect(actions[0].kind == ActionKind::kFold, "first legal action is fold");
  Expect(actions[1].kind == ActionKind::kCall, "second legal action is call");
  Expect(actions[1].amount == 1, "call amount is outstanding blind");
  Expect(actions[2].kind == ActionKind::kRaise, "third legal action is raise");
  Expect(actions[3].kind == ActionKind::kAllIn,
         "fourth legal action is all-in");
  Expect(actions[3].amount == 99, "all-in amount is the remaining stack");
}

void CheckLegalActionsApplyAndConserveChips(GameTree& tree) {
  std::vector<GameState> states = {PreflopState(), FlopState()};
  for (const GameState& state : states) {
    int total_chips = state.stack[0] + state.stack[1] + state.pot;
    std::vector<GameAction> actions = tree.get_legal_actions(state);
    Expect(!actions.empty(), "test states should have legal actions");
    for (const GameAction& action : actions) {
      GameState next = tree.apply_action(state, action);
      Expect(next.history.size() == state.history.size() + 1,
             "legal action should append to history");
      const GameAction& applied = next.history.back();
      Expect(applied.kind == action.kind,
             "applied legal action should keep action type");
      Expect(applied.player == state.player_to_act,
             "applied legal action should record acting player");
      Expect(next.stack[0] + next.stack[1] + next.pot == total_chips,
             "legal action should conserve chips");
    }
  }
}

void CheckAllInAction(GameTree& tree) {
  GameState all_in =
      tree.apply_action(PreflopState(), MakeAction(ActionKind::kAllIn));

  Expect(all_in.stack[0] == 0, "all-in commits the remaining stack");
  Expect(all_in.pot == 102, "all-in adds remaining chips to the pot");
  Expect(all_in.player_contribution[0] == 100,
         "all-in updates total player contribution");
  Expect(all_in.all_in, "all-in marks the state all-in");
  Expect(all_in.player_to_act == 1, "all-in passes action");
  Expect(all_in.history[0].kind == ActionKind::kAllIn, "all-in is recorded");
  Expect(all_in.history[0].amount == 99, "all-in records committed chips");
}

void CheckShortAllInCallClosesRound(GameTree& tree) {
  GameState state = FlopState();
  state.stack[0] = 5;

  GameState bet = tree.apply_action(state, MakeAction(ActionKind::kBet, 10));
  GameState called = tree.apply_action(bet, MakeAction(ActionKind::kCall));

  Expect(called.stack[0] == 0, "short all-in call commits caller stack");
  Expect(called.player_contribution[0] == 7,
         "short all-in call records partial contribution");
  Expect(called.player_contribution[1] == 12,
         "bettor keeps larger contribution");
  Expect(called.all_in, "short call marks state all-in");
  Expect(tree.is_betting_round_over(called),
         "short all-in call closes heads-up betting");
  Expect(tree.get_player_to_act(called) == -1,
         "short all-in call leaves no player action");

  GameTree::Node& root = tree.build_tree(called);
  Expect(root.is_chance_node, "short all-in call can run out the board");
}

void CheckAllInEquivalentActionsAreDeduped() {
  SolverConfig open_config;
  open_config.bet_sizes.push_back(24.5);
  GameTree open_tree(open_config);

  std::vector<GameAction> open_actions =
      open_tree.get_legal_actions(FlopState());
  Expect(open_actions.size() == 2,
         "full-stack bet size should become all-in only");
  Expect(open_actions[0].kind == ActionKind::kCheck,
         "first open action is check");
  Expect(open_actions[1].kind == ActionKind::kAllIn,
         "second open action is all-in");

  SolverConfig raise_config;
  raise_config.bet_sizes.push_back(32.7);
  GameTree raise_tree(raise_config);

  std::vector<GameAction> raise_actions =
      raise_tree.get_legal_actions(PreflopState());
  Expect(raise_actions.size() == 3,
         "full-stack raise size should become all-in only");
  Expect(raise_actions[0].kind == ActionKind::kFold,
         "first facing action is fold");
  Expect(raise_actions[1].kind == ActionKind::kCall,
         "second facing action is call");
  Expect(raise_actions[2].kind == ActionKind::kAllIn,
         "third facing action is all-in");
}

void CheckFullStackBetRaiseUseAllIn(GameTree& tree) {
  ExpectInvalidAction(tree, FlopState(), MakeAction(ActionKind::kBet, 98),
                      "full-stack bet should use all-in action");
  ExpectInvalidAction(tree, PreflopState(),
                      MakeAction(ActionKind::kRaise, 99),
                      "full-stack raise should use all-in action");
}

void CheckDuplicateConcreteBetSizesAreDeduped() {
  SolverConfig config;
  config.bet_sizes.push_back(0.5);
  config.bet_sizes.push_back(0.51);
  GameTree tree(config);

  std::vector<GameAction> open_actions = tree.get_legal_actions(FlopState());
  Expect(open_actions.size() == 3,
         "duplicate concrete bet sizes should collapse");
  Expect(open_actions[1].kind == ActionKind::kBet,
         "open action includes bet");
  Expect(open_actions[1].amount == 2, "duplicate bet amount is kept once");

  std::vector<GameAction> raise_actions =
      tree.get_legal_actions(PreflopState());
  Expect(raise_actions.size() == 4,
         "duplicate concrete raise sizes should collapse");
  Expect(raise_actions[2].kind == ActionKind::kRaise,
         "facing action includes raise");
  Expect(raise_actions[2].amount == 2,
         "duplicate raise amount is kept once");
}

void CheckStreetBetSizesOverrideGlobal() {
  SolverConfig config;
  config.bet_sizes.push_back(0.5);
  config.flop_bet_sizes.push_back(1.0);
  GameTree tree(config);

  std::vector<GameAction> preflop_actions =
      tree.get_legal_actions(PreflopState());
  Expect(preflop_actions[2].kind == ActionKind::kRaise,
         "global size should still apply preflop");
  Expect(preflop_actions[2].amount == 2,
         "preflop raise should use the global size");

  std::vector<GameAction> flop_actions = tree.get_legal_actions(FlopState());
  Expect(flop_actions[1].kind == ActionKind::kBet,
         "flop override should add a bet");
  Expect(flop_actions[1].amount == 4,
         "flop bet should use the street-specific size");
}

void CheckRaisesRemainAvailableAfterPriorRaise() {
  GameTree tree(TestConfig());

  GameState raised =
      tree.apply_action(PreflopState(), MakeAction(ActionKind::kRaise, 4));
  std::vector<GameAction> actions = tree.get_legal_actions(raised);

  Expect(actions.size() == 4, "facing action should include regular raise");
  Expect(actions[0].kind == ActionKind::kFold, "facing action keeps fold");
  Expect(actions[1].kind == ActionKind::kCall, "facing action keeps call");
  Expect(actions[2].kind == ActionKind::kRaise,
         "prior raise should not hide regular raise");
  Expect(actions[2].amount == 6,
         "regular raise amount should use configured pot fraction");
  Expect(actions[3].kind == ActionKind::kAllIn, "facing action keeps all-in");
}

void CheckActionKeyRejectsCollidingAmounts() {
  ExpectInvalidActionKey(MakeAction(ActionKind::kCall, -1),
                         "action key should reject negative amounts");
  ExpectInvalidActionKey(MakeAction(ActionKind::kCall, 1000000),
                         "action key should reject million-chip amounts");
}

void CheckShowdownUtility(GameTree& tree) {
  GameState showdown = ShowdownState();
  ComboId player_a =
      MakeCombo(14, SuitKind::kHearts, 14, SuitKind::kSpades);
  ComboId player_b =
      MakeCombo(13, SuitKind::kHearts, 13, SuitKind::kSpades);

  Expect(tree.is_terminal(showdown), "closed river action is terminal");
  Expect(tree.get_utility(showdown, player_a, player_b) == 10,
         "showdown utility is net chips for player A");
}

void CheckChanceAdvancesStreet(GameTree& tree) {
  GameState closed_preflop =
      tree.apply_action(PreflopState(), MakeAction(ActionKind::kCall));
  closed_preflop =
      tree.apply_action(closed_preflop, MakeAction(ActionKind::kCheck));

  GameTree::Node& root = tree.build_tree(closed_preflop);
  Expect(root.is_chance_node, "closed preflop action creates chance node");

  std::vector<CardId> flop = {
      MakeCardId(8, SuitKind::kHearts),
      MakeCardId(9, SuitKind::kClubs),
      MakeCardId(10, SuitKind::kSpades),
  };
  GameTree::Node& child_ref = tree.create_chance_child_node(root, 0, flop);

  Expect(child_ref.state.street == StreetKind::kFlop,
         "chance advances preflop to flop");
  Expect(child_ref.state.board_cards.size() == 3,
         "chance adds sampled flop cards");
  Expect(RankFromCardId(child_ref.state.board_cards[0]) == 8,
         "chance keeps sampled card order");
  Expect(child_ref.state.history.empty(),
         "chance clears street action history");
  Expect(child_ref.player_to_act == 1, "flop action starts with player 1");
  Expect(!child_ref.legal_actions.empty(),
         "chance child has legal player actions");
}

}  // namespace
}  // namespace poker

int main() {
  poker::GameTree tree(poker::TestConfig());
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
