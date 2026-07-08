#include "simulator.h"

#include "doctest/doctest.h"

namespace poker {
namespace {

NoLimitHoldemSimulator::Config TestConfig() {
  NoLimitHoldemSimulator::Config config;
  config.starting_stack = 100;
  config.small_blind = 1;
  config.big_blind = 2;
  return config;
}

Action MakeAction(ActionType type, int amount = 0) {
  Action action;
  action.set_action(type);
  action.set_amount(amount);
  return action;
}

TEST_CASE("simulator starts a deterministic hand with blinds posted") {
  NoLimitHoldemSimulator simulator(12345);
  BoardState state;
  Hand p0_hole;
  Hand p1_hole;

  REQUIRE(simulator.StartNewHand(TestConfig(), state, p0_hole, p1_hole));

  CHECK(p0_hole.cards_size() == 2);
  CHECK(p1_hole.cards_size() == 2);
  CHECK(state.street() == Street::PREFLOP);
  CHECK(state.cards_size() == 0);
  CHECK(state.stack_a() == 99);
  CHECK(state.stack_b() == 98);
  CHECK(state.pot() == 3);
  CHECK(state.player_contribution(0) == doctest::Approx(1.0));
  CHECK(state.player_contribution(1) == doctest::Approx(2.0));
  CHECK(state.player_to_act() == 0);
  CHECK(!simulator.IsTerminal(state));
}

TEST_CASE("call and check advance preflop to flop") {
  NoLimitHoldemSimulator simulator(12345);
  BoardState state;
  Hand p0_hole;
  Hand p1_hole;
  REQUIRE(simulator.StartNewHand(TestConfig(), state, p0_hole, p1_hole));

  REQUIRE(simulator.ApplyAction(MakeAction(ActionType::CALL, 1), state));
  CHECK(state.pot() == 4);
  CHECK(state.stack_a() == 98);
  CHECK(state.player_to_act() == 1);

  REQUIRE(simulator.ApplyAction(MakeAction(ActionType::CHECK), state));
  REQUIRE(simulator.AdvanceIfReady(state));
  CHECK(state.street() == Street::FLOP);
  CHECK(state.cards_size() == 3);
  CHECK(state.player_to_act() == 1);
  CHECK(!simulator.IsTerminal(state));
}

TEST_CASE("fold terminal showdown awards pot to opponent") {
  NoLimitHoldemSimulator simulator(12345);
  BoardState state;
  Hand p0_hole;
  Hand p1_hole;
  REQUIRE(simulator.StartNewHand(TestConfig(), state, p0_hole, p1_hole));

  REQUIRE(simulator.ApplyAction(MakeAction(ActionType::FOLD), state));
  CHECK(simulator.IsTerminal(state));

  const NoLimitHoldemSimulator::Result result =
      simulator.Showdown(state, p0_hole, p1_hole);
  CHECK(result.terminal);
  CHECK(result.winner == 1);
  CHECK(!result.split);
}

}  // namespace
}  // namespace poker
