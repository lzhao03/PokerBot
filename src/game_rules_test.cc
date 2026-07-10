#include "src/game_rules.h"

#include "doctest/doctest.h"
#include "src/combo.h"
#include "src/hand_evaluator.h"

#include <initializer_list>
#include <vector>

namespace poker {
namespace {

ExactGameState PreflopState() {
  ExactGameState state;
  state.betting.stack = {19, 18};
  state.betting.committed = {1, 2};
  return state;
}

ExactGameState FlopState() {
  ExactGameState state;
  state.betting.stack = {18, 18};
  state.betting.committed = {2, 2};
  state.betting.street = StreetKind::kFlop;
  state.betting.player_to_act = 1;
  state.board.add(MakeCardId(2, SuitKind::kHearts));
  state.board.add(MakeCardId(7, SuitKind::kDiamonds));
  state.board.add(MakeCardId(12, SuitKind::kClubs));
  return state;
}

ExactGameState RiverState() {
  ExactGameState state = FlopState();
  state.betting.stack = {20, 20};
  state.betting.committed = {10, 10};
  state.betting.street = StreetKind::kRiver;
  state.board.add(MakeCardId(9, SuitKind::kSpades));
  state.board.add(MakeCardId(3, SuitKind::kHearts));
  return state;
}

ExactGameState ApplySequence(ExactGameState state,
                             const std::vector<GameAction>& actions) {
  for (const GameAction& action : actions) {
    state.betting = ApplyAction(state.betting, action);
  }
  return state;
}

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

ExactGameState RiverShowdown(std::initializer_list<CardId> board) {
  ExactGameState state;
  state.betting.street = StreetKind::kRiver;
  state.betting.player_to_act = -1;
  state.betting.pending_action_mask = 0;
  state.betting.committed = {10, 10};
  for (CardId card : board) {
    state.board.add(card);
  }
  return state;
}

double OracleUtility(const ExactGameState& state,
                     ComboId a_hand,
                     ComboId b_hand,
                     int player) {
  const BettingState& betting = state.betting;
  const double committed = betting.committed[player];
  if (betting.folded_player == player) {
    return -committed;
  }
  if (betting.folded_player == Opponent(player)) {
    return Pot(betting) - committed;
  }

  HandEvaluator evaluator;
  const int comparison = evaluator.compare_hands(a_hand, b_hand, state.board);
  const int player_comparison = player == 0 ? comparison : -comparison;
  if (player_comparison > 0) {
    return Pot(betting) - committed;
  }
  if (player_comparison < 0) {
    return -committed;
  }
  return Pot(betting) / 2.0 - committed;
}

void CheckUtility(const ExactGameState& state,
                  ComboId a_hand,
                  ComboId b_hand,
                  double expected_a) {
  const double actual_a = GetUtility(state, a_hand, b_hand);
  const double oracle_a = OracleUtility(state, a_hand, b_hand, 0);
  const double oracle_b = OracleUtility(state, a_hand, b_hand, 1);
  CHECK(actual_a == doctest::Approx(expected_a));
  CHECK(actual_a == doctest::Approx(oracle_a));
  CHECK(oracle_a + oracle_b == doctest::Approx(0.0));
}

TEST_CASE("named betting sequences complete correctly") {
  struct CompletionCase {
    const char* name;
    ExactGameState initial;
    std::vector<GameAction> actions;
    bool terminal;
  };

  const std::vector<CompletionCase> cases = {
      {"preflop call-check", PreflopState(),
       {{ActionKind::kCall}, {ActionKind::kCheck}}, false},
      {"postflop check-check", FlopState(),
       {{ActionKind::kCheck}, {ActionKind::kCheck}}, false},
      {"bet-call", FlopState(),
       {{ActionKind::kBet, 4}, {ActionKind::kCall}}, false},
      {"bet-fold", FlopState(),
       {{ActionKind::kBet, 4}, {ActionKind::kFold}}, true},
      {"raise-call", PreflopState(),
       {{ActionKind::kRaise, 4}, {ActionKind::kCall}}, false},
      {"river check-check", RiverState(),
       {{ActionKind::kCheck}, {ActionKind::kCheck}}, true},
      {"river bet-call", RiverState(),
       {{ActionKind::kBet, 10}, {ActionKind::kCall}}, true},
      {"river bet-fold", RiverState(),
       {{ActionKind::kBet, 10}, {ActionKind::kFold}}, true},
  };

  for (const CompletionCase& test : cases) {
    CAPTURE(test.name);
    const ExactGameState result = ApplySequence(test.initial, test.actions);
    CHECK(IsBettingRoundOver(result.betting));
    CHECK(result.betting.player_to_act == -1);
    CHECK(IsTerminal(result.betting, result.board) == test.terminal);
  }
}

TEST_CASE("short all-in call completes with unmatched commitments") {
  ExactGameState state = PreflopState();
  state.betting.stack = {3, 12};
  state.betting.committed = {1, 8};
  state = ApplySequence(state, {{ActionKind::kCall}});

  CHECK(IsBettingRoundOver(state.betting));
  CHECK(state.betting.player_to_act == -1);
  CHECK(!IsTerminal(state.betting, state.board));
  CHECK(state.betting.stack[0] == 0);
  CHECK(state.betting.committed[0] < state.betting.committed[1]);
}

TEST_CASE("all-in call runs out the remaining board") {
  ExactGameState state = ApplySequence(
      PreflopState(), {{ActionKind::kCall}, {ActionKind::kCheck}});
  state = ApplyChance(state, {MakeCardId(2, SuitKind::kHearts),
                              MakeCardId(7, SuitKind::kDiamonds),
                              MakeCardId(12, SuitKind::kClubs)});
  CHECK(state.betting.street == StreetKind::kFlop);
  CHECK(state.board.count == 3);
  CHECK(state.betting.player_to_act == 1);

  state = ApplySequence(
      state, {{ActionKind::kAllIn}, {ActionKind::kCall}});
  CHECK(IsBettingRoundOver(state.betting));
  CHECK(state.betting.player_to_act == -1);

  state = ApplyChance(state, {MakeCardId(9, SuitKind::kSpades)});
  CHECK(state.betting.street == StreetKind::kTurn);
  CHECK(state.betting.player_to_act == -1);
  state = ApplyChance(state, {MakeCardId(3, SuitKind::kHearts)});
  CHECK(IsTerminal(state.betting, state.board));
  CHECK(state.betting.player_to_act == -1);
}

TEST_CASE("representative invalid actions are rejected") {
  const ExactGameState root = PreflopState();
  CHECK_THROWS(ApplyAction(root.betting, {ActionKind::kCheck}));
  CHECK_THROWS(ApplyAction(root.betting, {ActionKind::kRaise, 1}));
  CHECK_THROWS(ApplyAction(
      root.betting, {ActionKind::kRaise, root.betting.stack[0]}));

  const ExactGameState flop = FlopState();
  CHECK_THROWS(ApplyAction(flop.betting, {ActionKind::kCall}));
  CHECK_THROWS(ApplyAction(
      flop.betting, {ActionKind::kBet, flop.betting.stack[1]}));

  const ExactGameState folded =
      ApplySequence(root, {{ActionKind::kFold}});
  CHECK_THROWS(ApplyAction(folded.betting, {ActionKind::kCall}));

  ExactGameState broke = flop;
  broke.betting.stack[1] = 0;
  CHECK_THROWS(ApplyAction(broke.betting, {ActionKind::kCheck}));
}

TEST_CASE("non-sized action amounts are ignored") {
  const BettingState preflop = PreflopState().betting;
  CHECK(ApplyAction(preflop, {ActionKind::kCall}) ==
        ApplyAction(preflop, {ActionKind::kCall, 999}));
  CHECK(ApplyAction(preflop, {ActionKind::kFold}) ==
        ApplyAction(preflop, {ActionKind::kFold, 999}));

  const BettingState flop = FlopState().betting;
  CHECK(ApplyAction(flop, {ActionKind::kCheck}) ==
        ApplyAction(flop, {ActionKind::kCheck, 999}));
  CHECK(ApplyAction(flop, {ActionKind::kAllIn}) ==
        ApplyAction(flop, {ActionKind::kAllIn, 999}));
  CHECK_NOTHROW(ApplyAction(flop, {ActionKind::kFold, 999}));
}

TEST_CASE("terminal utility matches an independent oracle") {
  const ExactGameState folded =
      ApplySequence(PreflopState(), {{ActionKind::kFold}});
  CheckUtility(folded,
               Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts),
               Combo(12, SuitKind::kClubs, 11, SuitKind::kClubs), -1.0);

  CheckUtility(RiverShowdown({MakeCardId(14, SuitKind::kHearts),
                              MakeCardId(13, SuitKind::kHearts),
                              MakeCardId(12, SuitKind::kHearts),
                              MakeCardId(11, SuitKind::kHearts),
                              MakeCardId(2, SuitKind::kClubs)}),
               Combo(10, SuitKind::kHearts, 4, SuitKind::kClubs),
               Combo(9, SuitKind::kHearts, 5, SuitKind::kClubs), 10.0);

  CheckUtility(RiverShowdown({MakeCardId(2, SuitKind::kHearts),
                              MakeCardId(3, SuitKind::kDiamonds),
                              MakeCardId(4, SuitKind::kClubs),
                              MakeCardId(5, SuitKind::kSpades),
                              MakeCardId(6, SuitKind::kHearts)}),
               Combo(14, SuitKind::kClubs, 13, SuitKind::kDiamonds),
               Combo(12, SuitKind::kClubs, 11, SuitKind::kDiamonds), 0.0);
}

}  // namespace
}  // namespace poker
