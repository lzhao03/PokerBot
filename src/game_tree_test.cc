#include "src/game_tree.h"

#include <array>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Fn>
void ExpectThrows(Fn fn, const char* message) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

GameAction MakeAction(ActionKind kind, int amount = 0) {
  return {kind, amount, -1};
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

int TotalChips(const GameState& state) {
  return state.stack[0] + state.stack[1] + state.pot;
}

bool HasAction(const std::vector<GameAction>& actions,
               ActionKind kind,
               int amount = 0) {
  for (const GameAction& action : actions) {
    if (action.kind == kind && action.amount == amount) {
      return true;
    }
  }
  return false;
}

void CheckLegalActionsPreserveStateInvariants() {
  GameTree tree(TestConfig());
  std::vector<GameState> states;
  states.push_back(PreflopState());
  states.push_back(FlopState());
  states.push_back(tree.apply_action(PreflopState(),
                                     MakeAction(ActionKind::kRaise, 4)));
  states.push_back(tree.apply_action(FlopState(),
                                     MakeAction(ActionKind::kCheck)));

  for (const GameState& state : states) {
    const int total_chips = TotalChips(state);
    const std::vector<GameAction> actions = tree.get_legal_actions(state);
    Expect(!actions.empty(), "active state should have legal actions");
    for (const GameAction& action : actions) {
      const GameState next = tree.apply_action(state, action);
      Expect(TotalChips(next) == total_chips,
             "legal action should conserve chips");
      Expect(next.stack[0] >= 0 && next.stack[1] >= 0 && next.pot >= 0,
             "legal action should keep nonnegative chip counts");
      Expect(next.history.size() == state.history.size() + 1,
             "legal action should append history");
      Expect(next.history.back().kind == action.kind,
             "applied action should preserve action kind");
      Expect(next.history.back().player == state.player_to_act,
             "applied action should record acting player");
      if (next.folded_player >= 0) {
        Expect(tree.is_terminal(next), "folded state should be terminal");
        Expect(next.player_to_act == -1,
               "terminal fold should clear player to act");
      }
      if (tree.is_betting_round_over(next)) {
        Expect(tree.get_player_to_act(next) == -1,
               "closed betting round should have no player action");
      }
    }
  }
}

void CheckActionAbstractionShapes() {
  GameTree tree(TestConfig());
  const std::vector<GameAction> preflop =
      tree.get_legal_actions(PreflopState());
  Expect(HasAction(preflop, ActionKind::kFold),
         "facing blind should allow fold");
  Expect(HasAction(preflop, ActionKind::kCall, 1),
         "facing blind should allow call");
  Expect(HasAction(preflop, ActionKind::kRaise, 2),
         "configured raise should be legal");
  Expect(HasAction(preflop, ActionKind::kAllIn, 99),
         "facing blind should allow all-in");

  SolverConfig dedup_config;
  dedup_config.bet_sizes = {0.5, 0.51};
  GameTree dedup_tree(dedup_config);
  const std::vector<GameAction> dedup_actions =
      dedup_tree.get_legal_actions(FlopState());
  Expect(dedup_actions.size() == 3 &&
             HasAction(dedup_actions, ActionKind::kBet, 2),
         "duplicate concrete bet sizes should collapse");

  SolverConfig street_config;
  street_config.bet_sizes.push_back(0.5);
  street_config.flop_bet_sizes.push_back(1.0);
  GameTree street_tree(street_config);
  Expect(HasAction(street_tree.get_legal_actions(FlopState()),
                   ActionKind::kBet, 4),
         "street bet sizes should override global sizes");

  ExpectThrows([&] { (void)GameTree::action_key(MakeAction(ActionKind::kCall,
                                                           -1)); },
               "action key should reject negative amounts");
  ExpectThrows([&] { (void)GameTree::action_key(MakeAction(ActionKind::kCall,
                                                           1000000)); },
               "action key should reject colliding large amounts");
}

void CheckTerminalUtilityAndChance() {
  GameTree tree(TestConfig());
  GameState raised = tree.apply_action(PreflopState(),
                                       MakeAction(ActionKind::kRaise, 4));
  GameState folded = tree.apply_action(raised, MakeAction(ActionKind::kFold));
  Expect(tree.get_utility(folded, 0, 1) == 2,
         "fold utility should be net chips for player A");

  ComboId player_a =
      MakeCombo(14, SuitKind::kHearts, 14, SuitKind::kSpades);
  ComboId player_b =
      MakeCombo(13, SuitKind::kHearts, 13, SuitKind::kSpades);
  Expect(tree.is_terminal(ShowdownState()),
         "closed river action should be terminal");
  Expect(tree.get_utility(ShowdownState(), player_a, player_b) == 10,
         "showdown utility should score the best hand");

  GameState closed_preflop = tree.apply_action(
      tree.apply_action(PreflopState(), MakeAction(ActionKind::kCall)),
      MakeAction(ActionKind::kCheck));
  const std::array<CardId, 3> flop = {
      MakeCardId(8, SuitKind::kHearts),
      MakeCardId(9, SuitKind::kClubs),
      MakeCardId(10, SuitKind::kSpades),
  };
  const GameState child = tree.apply_chance(closed_preflop, flop);
  Expect(child.street == StreetKind::kFlop &&
             child.board_cards.size() == 3 &&
             child.history.empty() &&
             child.player_to_act == 1,
         "chance should advance street, add board, and reset action");
}

void CheckCompactHistoryCap() {
  GameTree tree(TestConfig());
  CompactPublicState state;
  state.stack = {99, 98};
  state.pot = 3;
  state.street = StreetKind::kPreflop;
  state.folded_player = -1;
  state.player_contribution = {1, 2};
  state.player_to_act = 0;
  for (int i = 0; i < CompactPublicState::kMaxHistoryActions; ++i) {
    AppendHistoryAction(state, {ActionKind::kCheck, 0, 0});
  }

  ExpectThrows([&] {
    (void)tree.apply_action(state, MakeAction(ActionKind::kCall));
  }, "compact action append should enforce history cap");
}

}  // namespace
}  // namespace poker

int main() {
  poker::CheckLegalActionsPreserveStateInvariants();
  poker::CheckActionAbstractionShapes();
  poker::CheckTerminalUtilityAndChance();
  poker::CheckCompactHistoryCap();
  return 0;
}
