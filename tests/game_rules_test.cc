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

TEST_CASE("short all-in calls refund unmatched chips") {
  ExactPublicState state = test::InitialHeadsUpState(4, 20, 1, 2);
  state.betting.stack = {3, 12};
  state.betting.total_committed = {1, 8};
  state.betting.street_committed = {1, 8};

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 8});

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
