#include "tests/rules_test_support.h"

#include "doctest/doctest.h"
#include "src/game_rules.h"

#include <array>

namespace poker {
namespace {

constexpr BettingRules kRules{2};

std::array<CardId, 3> Flop() {
  return {
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(12, SuitKind::kClubs),
  };
}

BettingState State(std::array<Chips, kPlayerCount> stack,
                   std::array<Chips, kPlayerCount> total,
                   std::array<Chips, kPlayerCount> street,
                   int player,
                   Chips last_full_raise) {
  BettingState state;
  state.stack = stack;
  state.total_committed = total;
  state.street_committed = street;
  state.player_to_act = static_cast<int8_t>(player);
  state.last_full_raise = last_full_raise;
  return state;
}

bool HasAction(const ActionMenu& menu, GameAction expected) {
  for (uint8_t i = 0; i < menu.count; ++i) {
    if (menu.actions[i] == expected) {
      return true;
    }
  }
  return false;
}

void CheckMenu(const BettingState& state,
               absl::Span<const double> sizes) {
  const ActionMenu menu = LegalActions(state, sizes);
  for (uint8_t i = 0; i < menu.count; ++i) {
    CHECK_NOTHROW(ApplyAction(state, menu.actions[i]));
  }
}

TEST_CASE("check-check completes a betting round") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 2, 2);
  state.betting.street = StreetKind::kFlop;
  state.betting.player_to_act = 1;
  state.betting.street_committed = {0, 0};
  state.board.deal_flop(Flop());

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CHECK_FALSE(IsBettingRoundOver(state.betting));
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});

  CHECK(IsBettingRoundOver(state.betting));
  CHECK(state.betting.player_to_act == -1);
  CHECK_FALSE(IsTerminal(state.betting, state.board));
}

TEST_CASE("commitments update and reset across streets") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  const std::array<Chips, kPlayerCount> chips = {
      state.betting.stack[0] + state.betting.total_committed[0],
      state.betting.stack[1] + state.betting.total_committed[1],
  };

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  CHECK(state.betting.stack[0] == 18);
  CHECK(state.betting.total_committed[0] == 2);
  CHECK(state.betting.street_committed[0] == 2);
  CHECK(Pot(state.betting) == 4);
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  state = ApplyChance(state, Flop(), kRules);

  CHECK(state.betting.total_committed ==
        std::array<Chips, kPlayerCount>{2, 2});
  CHECK(state.betting.street_committed ==
        std::array<Chips, kPlayerCount>{0, 0});
  CHECK(state.betting.last_full_raise == kRules.minimum_bet);

  state.betting = ApplyAction(state.betting, {ActionKind::kBet, 4});
  CHECK(state.betting.stack[1] == 14);
  CHECK(state.betting.total_committed[1] == 6);
  CHECK(state.betting.street_committed[1] == 4);
  CHECK(state.betting.last_full_raise == 4);
  CHECK(Pot(state.betting) == 8);
  for (size_t player = 0; player < kPlayerCount; ++player) {
    CHECK(state.betting.stack[player] +
              state.betting.total_committed[player] ==
          chips[player]);
  }
}

TEST_CASE("chip actions use final street commitments") {
  SUBCASE("raise") {
    ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
    state.betting = ApplyAction(state.betting, {ActionKind::kRaise, 5});

    CHECK(state.betting.stack ==
          std::array<Chips, kPlayerCount>{15, 18});
    CHECK(state.betting.total_committed ==
          std::array<Chips, kPlayerCount>{5, 2});
    CHECK(state.betting.street_committed ==
          std::array<Chips, kPlayerCount>{5, 2});
    CHECK(Pot(state.betting) == 7);
  }

  SUBCASE("all-in") {
    ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
    state.betting = ApplyAction(state.betting, {ActionKind::kAllIn, 20});

    CHECK(state.betting.stack ==
          std::array<Chips, kPlayerCount>{0, 18});
    CHECK(state.betting.total_committed ==
          std::array<Chips, kPlayerCount>{20, 2});
    CHECK(state.betting.street_committed ==
          std::array<Chips, kPlayerCount>{20, 2});
    CHECK(Pot(state.betting) == 22);
  }
}

TEST_CASE("minimum betting targets are enforced") {
  struct Case {
    const char* name;
    BettingState state;
    GameAction too_small;
    GameAction minimum;
  };
  const std::array<Case, 3> cases = {{
      {
          "preflop raise",
          State({19, 18}, {1, 2}, {1, 2}, 0, 2),
          {ActionKind::kRaise, 3},
          {ActionKind::kRaise, 4},
      },
      {
          "postflop opening bet",
          State({90, 90}, {10, 10}, {0, 0}, 0, 2),
          {ActionKind::kBet, 1},
          {ActionKind::kBet, 2},
      },
      {
          "raise after full raise",
          State({18, 14}, {2, 6}, {2, 6}, 0, 4),
          {ActionKind::kRaise, 9},
          {ActionKind::kRaise, 10},
      },
  }};

  for (const Case& test : cases) {
    CAPTURE(test.name);
    CHECK_THROWS_AS(ApplyAction(test.state, test.too_small),
                    std::invalid_argument);
    CHECK_NOTHROW(ApplyAction(test.state, test.minimum));
  }

  const BettingState preflop = cases[0].state;
  const std::array<double, 1> subminimum_size = {0.5};
  const ActionMenu menu = LegalActions(preflop, subminimum_size);
  CHECK_FALSE(HasAction(menu, {ActionKind::kRaise, 3}));
  CheckMenu(preflop, subminimum_size);
}

TEST_CASE("big blind retains the raise option after a limp") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  const std::array<double, 1> sizes = {0.5};
  const ActionMenu menu = LegalActions(state.betting, sizes);

  CHECK(HasAction(menu, {ActionKind::kCheck, 0}));
  CHECK(HasAction(menu, {ActionKind::kRaise, 4}));
  CHECK_FALSE(HasAction(menu, {ActionKind::kBet, 4}));
  CheckMenu(state.betting, sizes);
}

TEST_CASE("effective stacks and short all-ins bound aggression") {
  SUBCASE("full all-in raise") {
    const BettingState state =
        State({7, 20}, {8, 10}, {8, 10}, 0, 4);
    const BettingState child =
        ApplyAction(state, {ActionKind::kAllIn, 15});
    CHECK(child.street_committed[0] == 15);
    CHECK(child.last_full_raise == 5);
  }

  SUBCASE("short all-in raise") {
    const BettingState state =
        State({3, 20}, {8, 10}, {8, 10}, 0, 5);
    const std::array<double, 1> sizes = {1.0};
    const ActionMenu menu = LegalActions(state, sizes);
    CHECK(HasAction(menu, {ActionKind::kCall, 10}));
    CHECK(HasAction(menu, {ActionKind::kAllIn, 11}));
    CHECK_FALSE(HasAction(menu, {ActionKind::kRaise, 11}));

    const BettingState child =
        ApplyAction(state, {ActionKind::kAllIn, 11});
    CHECK(child.last_full_raise == 5);
    CHECK(child.pending_action_mask == PlayerBit(1));
    CheckMenu(state, sizes);
  }

  SUBCASE("deeper stack cannot exceed the effective stack") {
    const BettingState state =
        State({100, 20}, {10, 10}, {0, 0}, 0, 2);
    const std::array<double, 2> sizes = {0.5, 2.0};
    const ActionMenu menu = LegalActions(state, sizes);
    CHECK(HasAction(menu, {ActionKind::kAllIn, 20}));
    for (uint8_t i = 0; i < menu.count; ++i) {
      CHECK(menu.actions[i].target_street_commitment <= 20);
    }

    const BettingState child =
        ApplyAction(state, {ActionKind::kAllIn, 20});
    CHECK(child.stack[0] == 80);
    CheckMenu(state, sizes);
  }
}

TEST_CASE("an all-in opponent cannot face new aggression") {
  const BettingState settled =
      State({20, 0}, {10, 10}, {0, 0}, 0, 2);
  const std::array<double, 1> sizes = {1.0};
  const ActionMenu settled_menu = LegalActions(settled, sizes);
  CHECK(settled_menu.count == 1);
  CHECK(HasAction(settled_menu, {ActionKind::kCheck, 0}));

  const BettingState facing_bet =
      State({20, 0}, {0, 10}, {0, 10}, 0, 2);
  const ActionMenu call_menu = LegalActions(facing_bet, sizes);
  CHECK(HasAction(call_menu, {ActionKind::kFold, 0}));
  CHECK(HasAction(call_menu, {ActionKind::kCall, 10}));
  CHECK_FALSE(HasAction(call_menu, {ActionKind::kAllIn, 10}));

  const BettingState called =
      ApplyAction(facing_bet, {ActionKind::kCall, 10});
  CHECK(IsBettingRoundOver(called));
  CHECK(called.player_to_act == -1);
  CheckMenu(facing_bet, sizes);
}

TEST_CASE("preflop all-in runout skips later decisions") {
  ExactPublicState state = test::InitialHeadsUpState(4, 20, 1, 2);
  state.betting = ApplyAction(state.betting, {ActionKind::kAllIn, 4});
  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 4});
  REQUIRE(state.betting.player_to_act == -1);

  state = ApplyChance(state, Flop(), kRules);
  CHECK(state.betting.player_to_act == -1);
  CHECK_FALSE(IsTerminal(state.betting, state.board));

  const std::array<CardId, 1> turn = {
      MakeCardId(9, SuitKind::kSpades),
  };
  state = ApplyChance(state, turn, kRules);
  CHECK(state.betting.player_to_act == -1);
  CHECK_FALSE(IsTerminal(state.betting, state.board));

  const std::array<CardId, 1> river = {
      MakeCardId(3, SuitKind::kHearts),
  };
  state = ApplyChance(state, river, kRules);
  CHECK(state.betting.player_to_act == -1);
  CHECK(IsTerminal(state.betting, state.board));
}

TEST_CASE("short all-in calls refund unmatched chips") {
  ExactPublicState state = test::InitialHeadsUpState(4, 20, 1, 2);
  state.betting.stack = {3, 12};
  state.betting.total_committed = {1, 8};
  state.betting.street_committed = {1, 8};

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 4});

  CHECK(state.betting.total_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(state.betting.street_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(state.betting.stack ==
        std::array<Chips, kPlayerCount>{0, 16});
  CHECK(IsValidBettingState(state.betting));
}

TEST_CASE("fold completes the hand") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  state.betting = ApplyAction(state.betting, {ActionKind::kFold});

  CHECK(IsBettingRoundOver(state.betting));
  CHECK(IsTerminal(state.betting, state.board));
  CHECK(state.betting.folded_player == 0);
  CHECK(state.betting.player_to_act == -1);
}

}  // namespace
}  // namespace poker
