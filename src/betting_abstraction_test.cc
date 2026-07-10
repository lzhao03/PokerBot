#include "src/betting_abstraction.h"

#include "doctest/doctest.h"

#include "src/game_rules.h"

namespace poker {
namespace {

BettingState FlopState() {
  BettingState state;
  state.stack[0] = 98;
  state.stack[1] = 98;
  state.street = StreetKind::kFlop;
  state.folded_player = -1;
  state.committed = {2, 2};
  state.player_to_act = 1;
  return state;
}

bool HasAction(const ActionMenu& menu,
               ActionKind kind,
               int amount = 0) {
  for (uint8_t i = 0; i < menu.count; ++i) {
    const GameAction& action = menu.actions[i];
    if (action.kind == kind && action.amount == amount) {
      return true;
    }
  }
  return false;
}

TEST_CASE("street-specific bet sizes override defaults") {
  SolverConfig config;
  config.bet_sizes = {0.5};
  config.flop_bet_sizes = {1.0};
  BettingAbstraction betting(config);

  const ActionMenu menu = betting.actions_for_betting_node(FlopState());
  CHECK(HasAction(menu, ActionKind::kBet, 4));
  CHECK_FALSE(HasAction(menu, ActionKind::kBet, 2));
}

TEST_CASE("bet sizes use exact pot and stack values") {
  SolverConfig config;
  config.bet_sizes = {0.5};
  BettingAbstraction betting(config);

  BettingState state;
  state.stack = {200, 200};
  state.committed = {50, 50};
  state.street = StreetKind::kFlop;
  state.player_to_act = 0;
  state.folded_player = -1;

  const ActionMenu menu = betting.actions_for_betting_node(state);
  CHECK(HasAction(menu, ActionKind::kBet, 50));
  CHECK(HasAction(menu, ActionKind::kAllIn, 200));
}

}  // namespace
}  // namespace poker
