#include "tests/rules_test_support.h"

#include "doctest/doctest.h"
#include "src/game_rules.h"

#include <array>

namespace poker {
namespace {

TEST_CASE("check-check completes a betting round") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 2, 2);
  state.betting.street = StreetKind::kFlop;
  state.betting.player_to_act = 1;
  const std::array<CardId, 3> flop = {
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(12, SuitKind::kClubs),
  };
  state.board.deal_flop(flop);

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CHECK_FALSE(IsBettingRoundOver(state.betting));
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});

  CHECK(IsBettingRoundOver(state.betting));
  CHECK(state.betting.player_to_act == -1);
  CHECK_FALSE(IsTerminal(state.betting, state.board));
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
