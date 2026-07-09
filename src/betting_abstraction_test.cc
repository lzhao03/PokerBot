#include "src/betting_abstraction.h"

#include "doctest/doctest.h"

#include <array>
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

CompactPublicState PreflopState() {
  CompactPublicState state;
  state.stack[0] = 99;
  state.stack[1] = 98;
  state.pot = 3;
  state.street = StreetKind::kPreflop;
  state.folded_player = -1;
  state.player_contribution = {1, 2};
  state.player_to_act = 0;
  return state;
}

CompactPublicState FlopState() {
  CompactPublicState state;
  state.stack[0] = 98;
  state.stack[1] = 98;
  state.pot = 4;
  state.street = StreetKind::kFlop;
  state.folded_player = -1;
  state.player_contribution = {2, 2};
  state.player_to_act = 1;
  return state;
}

int TotalChips(const CompactPublicState& state) {
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

std::vector<GameAction> LegalActions(const BettingAbstraction& betting,
                                     const CompactPublicState& state) {
  std::array<GameAction, kMaxActionsPerNode> action_table = {};
  const uint8_t action_count = betting.actions_for_betting_node(
      state, state.player_to_act, action_table);
  return std::vector<GameAction>(action_table.begin(),
                                 action_table.begin() + action_count);
}

TEST_CASE("generated betting actions preserve state invariants") {
  GameTree tree;
  BettingAbstraction betting(TestConfig());
  const std::vector<CompactPublicState> states = {
      PreflopState(),
      FlopState(),
      tree.apply_action(PreflopState(), {ActionKind::kRaise, 4, -1}),
      tree.apply_action(FlopState(), {ActionKind::kCheck, 0, -1}),
  };

  for (const CompactPublicState& state : states) {
    const int total_chips = TotalChips(state);
    const std::vector<GameAction> actions = LegalActions(betting, state);
    REQUIRE(!actions.empty());
    for (const GameAction& action : actions) {
      CAPTURE(action.kind);
      CAPTURE(action.amount);
      const CompactPublicState next = tree.apply_action(state, action);
      CHECK(TotalChips(next) == total_chips);
      CHECK(next.stack[0] >= 0);
      CHECK(next.stack[1] >= 0);
      CHECK(next.pot >= 0);
      CHECK(next.history_size == state.history_size + 1);
      const GameAction last_action = MakeGameAction(next.last_action);
      CHECK(last_action.kind == action.kind);
      CHECK(last_action.player == state.player_to_act);
      if (next.folded_player >= 0) {
        CHECK(tree.is_terminal(next));
        CHECK(next.player_to_act == -1);
      }
      if (tree.is_betting_round_over(next)) {
        CHECK(tree.get_player_to_act(next) == -1);
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

  CHECK_THROWS((void)betting.action_key({ActionKind::kCall, -1, -1}));
  CHECK_THROWS((void)betting.action_key({ActionKind::kCall, 1000000, -1}));
}

}  // namespace
}  // namespace poker
