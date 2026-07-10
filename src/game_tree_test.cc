#include "src/game_tree.h"

#include "doctest/doctest.h"

#include <array>

namespace poker {
namespace {

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

CompactPublicState ShowdownState() {
  CompactPublicState state;
  state.stack[0] = 90;
  state.stack[1] = 90;
  state.pot = 20;
  state.street = StreetKind::kRiver;
  state.folded_player = -1;
  state.player_contribution = {10, 10};
  state.player_to_act = 1;
  state.actions_this_street = 2;
  state.last_action = {ActionKind::kCheck, 0, 0};
  AddBoardCard(state, MakeCardId(2, SuitKind::kHearts));
  AddBoardCard(state, MakeCardId(7, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(9, SuitKind::kClubs));
  AddBoardCard(state, MakeCardId(11, SuitKind::kSpades));
  AddBoardCard(state, MakeCardId(12, SuitKind::kDiamonds));
  return state;
}

CompactPublicState CompactApplyChance(const CompactPublicState& state,
                                      absl::Span<const CardId> cards) {
  return ToCompact(ApplyChance(ExactGameStateFromCompact(state), cards));
}

TEST_CASE("terminal utility and chance transitions are correct") {
  CompactPublicState raised = ApplyAction(
      PreflopState(), {ActionKind::kRaise, 4, -1});
  CompactPublicState folded = ApplyAction(
      raised, {ActionKind::kFold, 0, -1});
  CHECK(GetUtility(ExactGameStateFromCompact(folded), 0, 1) == 2);

  ComboId player_a =
      CardsToComboId(MakeCardId(14, SuitKind::kHearts),
                     MakeCardId(14, SuitKind::kSpades));
  ComboId player_b =
      CardsToComboId(MakeCardId(13, SuitKind::kHearts),
                     MakeCardId(13, SuitKind::kSpades));
  const ExactGameState showdown = ExactGameStateFromCompact(ShowdownState());
  CHECK(IsTerminal(showdown.betting, showdown.board));
  CHECK(GetUtility(ExactGameStateFromCompact(ShowdownState()), player_a,
                   player_b) == 10);

  CompactPublicState closed_preflop = ApplyAction(
      ApplyAction(PreflopState(), {ActionKind::kCall, 0, -1}),
      {ActionKind::kCheck, 0, -1});
  const std::array<CardId, 3> flop = {
      MakeCardId(8, SuitKind::kHearts),
      MakeCardId(9, SuitKind::kClubs),
      MakeCardId(10, SuitKind::kSpades),
  };
  const CompactPublicState child = CompactApplyChance(closed_preflop, flop);
  CHECK(child.street == StreetKind::kFlop);
  CHECK(child.board_count == 3);
  CHECK(child.actions_this_street == 0);
  CHECK(child.player_to_act == 1);
}

}  // namespace
}  // namespace poker
