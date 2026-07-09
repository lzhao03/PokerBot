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
  AppendHistoryAction(state, {ActionKind::kCheck, 0, 1});
  AppendHistoryAction(state, {ActionKind::kCheck, 0, 0});
  AddBoardCard(state, MakeCardId(2, SuitKind::kHearts));
  AddBoardCard(state, MakeCardId(7, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(9, SuitKind::kClubs));
  AddBoardCard(state, MakeCardId(11, SuitKind::kSpades));
  AddBoardCard(state, MakeCardId(12, SuitKind::kDiamonds));
  return state;
}

TEST_CASE("terminal utility and chance transitions are correct") {
  GameTree tree;
  CompactPublicState raised = tree.apply_action(
      PreflopState(), {ActionKind::kRaise, 4, -1});
  CompactPublicState folded = tree.apply_action(
      raised, {ActionKind::kFold, 0, -1});
  CHECK(tree.get_utility(folded, 0, 1) == 2);

  ComboId player_a =
      CardsToComboId(MakeCardId(14, SuitKind::kHearts),
                     MakeCardId(14, SuitKind::kSpades));
  ComboId player_b =
      CardsToComboId(MakeCardId(13, SuitKind::kHearts),
                     MakeCardId(13, SuitKind::kSpades));
  CHECK(tree.is_terminal(ShowdownState()));
  CHECK(tree.get_utility(ShowdownState(), player_a, player_b) == 10);

  CompactPublicState closed_preflop = tree.apply_action(
      tree.apply_action(PreflopState(), {ActionKind::kCall, 0, -1}),
      {ActionKind::kCheck, 0, -1});
  const std::array<CardId, 3> flop = {
      MakeCardId(8, SuitKind::kHearts),
      MakeCardId(9, SuitKind::kClubs),
      MakeCardId(10, SuitKind::kSpades),
  };
  const CompactPublicState child = tree.apply_chance(closed_preflop, flop);
  CHECK(child.street == StreetKind::kFlop);
  CHECK(child.board_count == 3);
  CHECK(child.history_size == 0);
  CHECK(child.player_to_act == 1);
}

TEST_CASE("compact history cap is enforced") {
  GameTree tree;
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

  CHECK_THROWS((void)tree.apply_action(state, {ActionKind::kCall, 0, -1}));
}

}  // namespace
}  // namespace poker
