#include "src/poker.h"

#include "doctest/doctest.h"

#include <array>
#include <stdexcept>
#include <string>

namespace poker {
namespace {

using S = SuitKind;

Card C(int rank, S suit) { return MakeCardId(rank, suit); }

BettingState Apply(const BettingState& state, GameAction action) {
  const auto child = TryApplyAction(state, action);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

ExactPublicState Root() {
  ExactPublicState state;
  state.betting.stack = {19, 18};
  state.betting.total_committed = {1, 2};
  state.betting.street_committed = {1, 2};
  state.betting.last_full_raise = 2;
  return state;
}

constexpr BettingRules kRules{2};

TEST_CASE("boundary actions, chance transitions, and sizing are enforced") {
  ExactPublicState state = Root();
  CHECK_THROWS(Apply(state.betting, {ActionKind::kCheck}));
  CHECK_THROWS(Apply(state.betting, {ActionKind::kRaise, 1}));
  state.betting = Apply(state.betting, {ActionKind::kCall, 2});
  state.betting = Apply(state.betting, {ActionKind::kCheck});
  CHECK_THROWS_AS(ApplyChance(state, {C(14, S::kSpades)}, kRules),
                  std::invalid_argument);

  ExactPublicState short_call = Root();
  short_call.betting.stack = {3, 12};
  short_call.betting.total_committed = {1, 8};
  short_call.betting.street_committed = {1, 8};
  short_call.betting = Apply(short_call.betting,
                                   {ActionKind::kCall, 4});
  CHECK(IsBettingRoundOver(short_call.betting));
  CHECK(short_call.betting.stack[0] == 0);
  CHECK(short_call.betting.total_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(short_call.betting.street_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(short_call.betting.stack[1] == 16);

  SolverConfig config;
  config.bet_sizes[static_cast<size_t>(StreetKind::kPreflop)] = {0.5};
  config.bet_sizes[static_cast<size_t>(StreetKind::kFlop)] = {1.0};
  BettingState flop;
  flop.stack = {98, 98};
  flop.total_committed = {2, 2};
  flop.last_full_raise = 2;
  flop.street = StreetKind::kFlop;
  flop.player_to_act = 1;
  flop.folded_player = -1;
  const SolverTransitions transitions = GenerateTransitions(config, flop);
  bool bet_four = false;
  bool bet_two = false;
  bool all_in = false;
  for (const SolverTransition& transition : transitions) {
    const GameAction action = transition.action;
    bet_four |= action == GameAction{ActionKind::kBet, 4};
    bet_two |= action == GameAction{ActionKind::kBet, 2};
    all_in |= action == GameAction{ActionKind::kAllIn, 98};
  }
  CHECK(bet_four);
  CHECK_FALSE(bet_two);
  CHECK(all_in);
}

}  // namespace
}  // namespace poker
