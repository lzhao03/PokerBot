#include "src/game_tree.h"

#include "doctest/doctest.h"

#include <array>

namespace poker {
namespace {

ExactGameState PreflopState() {
  ExactGameState state;
  state.betting.stack = {99, 98};
  state.betting.street = StreetKind::kPreflop;
  state.betting.folded_player = -1;
  state.betting.committed = {1, 2};
  state.betting.player_to_act = 0;
  return state;
}

ExactGameState ShowdownState() {
  ExactGameState state;
  state.betting.stack = {90, 90};
  state.betting.street = StreetKind::kRiver;
  state.betting.folded_player = -1;
  state.betting.committed = {10, 10};
  state.betting.player_to_act = -1;
  state.betting.pending_action_mask = 0;
  state.board.add(MakeCardId(2, SuitKind::kHearts));
  state.board.add(MakeCardId(7, SuitKind::kDiamonds));
  state.board.add(MakeCardId(9, SuitKind::kClubs));
  state.board.add(MakeCardId(11, SuitKind::kSpades));
  state.board.add(MakeCardId(12, SuitKind::kDiamonds));
  return state;
}

ExactGameState ApplyStateAction(ExactGameState state,
                                const GameAction& action) {
  state.betting = ApplyAction(state.betting, action);
  return state;
}

TEST_CASE("terminal utility and chance transitions are correct") {
  ExactGameState raised = ApplyStateAction(
      PreflopState(), {ActionKind::kRaise, 4});
  ExactGameState folded = ApplyStateAction(
      raised, {ActionKind::kFold, 0});
  CHECK(GetUtility(folded, 0, 1) == 2);

  ComboId player_a =
      CardsToComboId(MakeCardId(14, SuitKind::kHearts),
                     MakeCardId(14, SuitKind::kSpades));
  ComboId player_b =
      CardsToComboId(MakeCardId(13, SuitKind::kHearts),
                     MakeCardId(13, SuitKind::kSpades));
  const ExactGameState showdown = ShowdownState();
  CHECK(IsTerminal(showdown.betting, showdown.board));
  CHECK(GetUtility(showdown, player_a, player_b) == 10);

  ExactGameState closed_preflop = ApplyStateAction(
      ApplyStateAction(PreflopState(), {ActionKind::kCall, 0}),
      {ActionKind::kCheck, 0});
  const std::array<CardId, 3> flop = {
      MakeCardId(8, SuitKind::kHearts),
      MakeCardId(9, SuitKind::kClubs),
      MakeCardId(10, SuitKind::kSpades),
  };
  const ExactGameState child = ApplyChance(closed_preflop, flop);
  CHECK(child.betting.street == StreetKind::kFlop);
  CHECK(child.board.count == 3);
  CHECK(child.betting.pending_action_mask == kAllPlayersMask);
  CHECK(child.betting.player_to_act == 1);
}

}  // namespace
}  // namespace poker
