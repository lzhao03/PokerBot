#include "src/betting_abstraction.h"

#include "doctest/doctest.h"

#include <vector>

#include "src/game_tree.h"

namespace poker {
namespace {

SolverConfig TestConfig() {
  SolverConfig config;
  config.bet_sizes.push_back(0.5);
  config.starting_stack_size = 100;
  config.small_blind = 1;
  config.big_blind = 2;
  return config;
}

BettingState PreflopState() {
  BettingState state;
  state.stack[0] = 99;
  state.stack[1] = 98;
  state.pot = 3;
  state.street = StreetKind::kPreflop;
  state.folded_player = -1;
  state.committed = {1, 2};
  state.player_to_act = 0;
  return state;
}

BettingState FlopState() {
  BettingState state;
  state.stack[0] = 98;
  state.stack[1] = 98;
  state.pot = 4;
  state.street = StreetKind::kFlop;
  state.folded_player = -1;
  state.committed = {2, 2};
  state.player_to_act = 1;
  return state;
}

int TotalChips(const BettingState& state) {
  return state.stack[0] + state.stack[1] + Pot(state);
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

std::vector<GameAction> LegalActions(const BettingAbstraction& betting,
                                     const BettingState& state) {
  const auto menu =
      betting.actions_for_betting_node(state, state.player_to_act);
  return std::vector<GameAction>(menu.actions.begin(),
                                 menu.actions.begin() + menu.count);
}

TEST_CASE("generated betting actions preserve state invariants") {
  BettingAbstraction betting(TestConfig());
  const std::vector<BettingState> states = {
      PreflopState(),
      FlopState(),
      ApplyAction(PreflopState(), {ActionKind::kRaise, 4, -1}),
      ApplyAction(FlopState(), {ActionKind::kCheck, 0, -1}),
  };

  for (const BettingState& state : states) {
    const int total_chips = TotalChips(state);
    const std::vector<GameAction> actions = LegalActions(betting, state);
    REQUIRE(!actions.empty());
    for (const GameAction& action : actions) {
      CAPTURE(action.kind);
      CAPTURE(action.amount);
      const BettingState next = ApplyAction(state, action);
      CHECK(TotalChips(next) == total_chips);
      CHECK(next.stack[0] >= 0);
      CHECK(next.stack[1] >= 0);
      CHECK(next.pot >= 0);
      CHECK(next.actions_this_street == state.actions_this_street + 1);
      CHECK(next.last_action.kind == action.kind);
      CHECK(next.last_action.player == state.player_to_act);
      if (next.folded_player >= 0) {
        CHECK(IsTerminal(next, Board{}));
        CHECK(next.player_to_act == -1);
      }
      if (IsBettingRoundOver(next)) {
        CHECK(GetPlayerToAct(next, Board{}) == -1);
      }
    }
  }
}

TEST_CASE("betting action menu follows config") {
  BettingAbstraction betting(TestConfig());
  const std::vector<GameAction> preflop =
      LegalActions(betting, PreflopState());
  CHECK(HasAction(preflop, ActionKind::kFold));
  CHECK(HasAction(preflop, ActionKind::kCall, 1));
  CHECK(HasAction(preflop, ActionKind::kRaise, 2));
  CHECK(HasAction(preflop, ActionKind::kAllIn, 99));

  SolverConfig dedup_config;
  dedup_config.bet_sizes = {0.5, 0.51};
  BettingAbstraction dedup_betting(dedup_config);
  const std::vector<GameAction> dedup_actions =
      LegalActions(dedup_betting, FlopState());
  CHECK(dedup_actions.size() == 3);
  CHECK(HasAction(dedup_actions, ActionKind::kBet, 2));

  SolverConfig street_config;
  street_config.bet_sizes.push_back(0.5);
  street_config.flop_bet_sizes.push_back(1.0);
  BettingAbstraction street_betting(street_config);
  CHECK(HasAction(LegalActions(street_betting, FlopState()), ActionKind::kBet,
                  4));
}

TEST_CASE("sized betting actions are sorted and deduplicated") {
  SolverConfig config;
  config.bet_sizes = {1.0, 0.5, 0.51, 0.25};
  BettingAbstraction betting(config);

  const std::vector<GameAction> actions = LegalActions(betting, FlopState());

  REQUIRE(actions.size() == 5);
  CHECK(actions[0].kind == ActionKind::kCheck);
  CHECK(actions[1].kind == ActionKind::kBet);
  CHECK(actions[1].amount == 1);
  CHECK(actions[2].kind == ActionKind::kBet);
  CHECK(actions[2].amount == 2);
  CHECK(actions[3].kind == ActionKind::kBet);
  CHECK(actions[3].amount == 4);
  CHECK(actions[4].kind == ActionKind::kAllIn);
}

TEST_CASE("bet sizes use exact pot and stack values") {
  SolverConfig config;
  config.bet_sizes = {0.5};
  BettingAbstraction betting(config);

  BettingState state;
  state.pot = 100;
  state.stack = {200, 200};
  state.committed = {0, 0};
  state.street = StreetKind::kFlop;
  state.player_to_act = 0;
  state.folded_player = -1;

  const auto menu = betting.actions_for_betting_node(state, 0);
  const std::vector<GameAction> actions(menu.actions.begin(),
                                        menu.actions.begin() + menu.count);
  CHECK(HasAction(actions, ActionKind::kBet, 50));
  CHECK(HasAction(actions, ActionKind::kAllIn, 200));
}

}  // namespace
}  // namespace poker
