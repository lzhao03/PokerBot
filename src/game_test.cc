#include "src/betting_abstraction.h"
#include "src/card_utils.h"
#include "src/combo.h"
#include "src/game_rules.h"
#include "src/hand_evaluator.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

CardId Card(int rank, SuitKind suit) {
  return MakeCardId(rank, suit);
}

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(Card(first_rank, first_suit),
                        Card(second_rank, second_suit));
}

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
  state.board.add(Card(2, SuitKind::kHearts));
  state.board.add(Card(7, SuitKind::kDiamonds));
  state.board.add(Card(12, SuitKind::kClubs));
  return state;
}

ExactGameState RiverState() {
  ExactGameState state = FlopState();
  state.betting.stack = {20, 20};
  state.betting.committed = {10, 10};
  state.betting.street = StreetKind::kRiver;
  state.board.add(Card(9, SuitKind::kSpades));
  state.board.add(Card(3, SuitKind::kHearts));
  return state;
}

ExactGameState ApplySequence(ExactGameState state,
                             std::initializer_list<GameAction> actions) {
  for (const GameAction& action : actions) {
    state.betting = ApplyAction(state.betting, action);
  }
  return state;
}

bool HasAction(const ActionMenu& menu, ActionKind kind, int amount = 0) {
  for (uint8_t i = 0; i < menu.count; ++i) {
    if (menu.actions[i].kind == kind && menu.actions[i].amount == amount) {
      return true;
    }
  }
  return false;
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

int TotalChips(const ExactGameState& state) {
  return Pot(state.betting) + state.betting.stack[0] +
         state.betting.stack[1];
}

void CheckState(const ExactGameState& state, int total_chips) {
  const BettingState& betting = state.betting;
  RC_ASSERT(IsValidBettingState(betting));
  RC_ASSERT(TotalChips(state) == total_chips);
  RC_ASSERT(Pot(betting) == betting.committed[0] + betting.committed[1]);
  RC_ASSERT(std::popcount(state.board.mask) == state.board.count);
  RC_ASSERT(BoardCardsForStreet(betting.street) == state.board.count);
  RC_ASSERT(IsBettingRoundOver(betting) == !IsPlayer(betting.player_to_act));

  CardMask seen = 0;
  for (CardId card : state.board.span()) {
    RC_ASSERT((seen & CardBit(card)) == 0);
    RC_ASSERT(state.board.contains(card));
    seen |= CardBit(card);
  }
}

void CheckActionTransition(const ExactGameState& parent,
                           const ExactGameState& child,
                           int player,
                           int total_chips) {
  CheckState(child, total_chips);
  const Chips committed =
      parent.betting.stack[player] - child.betting.stack[player];
  RC_ASSERT(committed >= 0);
  RC_ASSERT(Pot(child.betting) == Pot(parent.betting) + committed);
  RC_ASSERT(child.betting.committed[player] ==
            parent.betting.committed[player] + committed);
  RC_ASSERT(child.betting.committed[Opponent(player)] ==
            parent.betting.committed[Opponent(player)]);
  RC_ASSERT(child.board.mask == parent.board.mask);
}

ActionMenu CheckLegalMenu(const ExactGameState& state,
                          absl::Span<const double> bet_sizes,
                          int total_chips) {
  const BettingState& betting = state.betting;
  const ActionMenu menu = LegalActions(betting, bet_sizes);
  RC_ASSERT(menu.count > 0);

  std::vector<double> sorted_sizes(bet_sizes.begin(), bet_sizes.end());
  std::sort(sorted_sizes.begin(), sorted_sizes.end());
  const ActionMenu sorted_menu = LegalActions(betting, sorted_sizes);
  RC_ASSERT(menu.count == sorted_menu.count);

  for (uint8_t i = 0; i < menu.count; ++i) {
    const GameAction action = menu.actions[i];
    RC_ASSERT(action == sorted_menu.actions[i]);
    for (uint8_t j = 0; j < i; ++j) {
      RC_ASSERT(action != menu.actions[j]);
    }
    ExactGameState child = state;
    child.betting = ApplyAction(betting, action);
    CheckActionTransition(state, child, betting.player_to_act, total_chips);
  }

  uint8_t index = 0;
  const Chips to_call = ToCall(betting, betting.player_to_act);
  if (to_call > 0) {
    RC_ASSERT(menu.actions[index++] == GameAction{ActionKind::kFold});
    RC_ASSERT((menu.actions[index++] ==
               GameAction{ActionKind::kCall,
                          std::min(to_call,
                                   betting.stack[betting.player_to_act])}));
  } else {
    RC_ASSERT(menu.actions[index++] == GameAction{ActionKind::kCheck});
  }

  const ActionKind sized_kind =
      to_call > 0 ? ActionKind::kRaise : ActionKind::kBet;
  Chips previous_amount = 0;
  while (index < menu.count && menu.actions[index].kind == sized_kind) {
    RC_ASSERT(menu.actions[index].amount > previous_amount);
    previous_amount = menu.actions[index++].amount;
  }
  if (index < menu.count) {
    RC_ASSERT((menu.actions[index++] ==
               GameAction{ActionKind::kAllIn,
                          betting.stack[betting.player_to_act]}));
  }
  RC_ASSERT(index == menu.count);
  return menu;
}

CardId SelectAvailableCard(CardMask blocked, uint16_t choice) {
  std::array<CardId, kDeckCardCount> available = {};
  size_t count = 0;
  for (int id = 0; id < kDeckCardCount; ++id) {
    const CardId card = static_cast<CardId>(id);
    if ((blocked & CardBit(card)) == 0) {
      available[count++] = card;
    }
  }
  RC_ASSERT(count > 0);
  return available[choice % count];
}

void CheckChanceTransition(const ExactGameState& parent,
                           const ExactGameState& child,
                           const std::vector<CardId>& cards,
                           int total_chips) {
  CheckState(child, total_chips);
  RC_ASSERT(child.betting.stack == parent.betting.stack);
  RC_ASSERT(child.betting.committed == parent.betting.committed);
  RC_ASSERT(child.betting.street == StreetAfterChance(parent.betting.street));
  RC_ASSERT(child.betting.pending_action_mask == kAllPlayersMask);
  RC_ASSERT((child.board.mask & parent.board.mask) == parent.board.mask);

  CardMask dealt = 0;
  for (CardId card : cards) {
    RC_ASSERT((parent.board.mask & CardBit(card)) == 0);
    RC_ASSERT((dealt & CardBit(card)) == 0);
    dealt |= CardBit(card);
  }
}

void ReplayAndValidate(int starting_stack,
                       const std::vector<uint8_t>& size_points,
                       const std::array<uint16_t, 64>& choices) {
  std::vector<double> bet_sizes;
  bet_sizes.reserve(size_points.size());
  for (uint8_t points : size_points) {
    bet_sizes.push_back((static_cast<double>(points) + 1.0) / 100.0);
  }

  ExactGameState state;
  state.betting.stack = {starting_stack - 1, starting_stack - 2};
  state.betting.committed = {1, 2};
  const int total_chips = TotalChips(state);
  size_t cursor = 0;

  while (true) {
    CheckState(state, total_chips);
    if (IsTerminal(state.betting, state.board) || cursor == choices.size()) {
      return;
    }

    if (IsPlayer(state.betting.player_to_act)) {
      const ActionMenu menu = CheckLegalMenu(state, bet_sizes, total_chips);
      state.betting =
          ApplyAction(state.betting, menu.actions[choices[cursor++] % menu.count]);
      continue;
    }

    const int card_count = CardsForNextStreet(state.betting.street);
    if (cursor + static_cast<size_t>(card_count) > choices.size()) {
      return;
    }
    std::vector<CardId> cards;
    cards.reserve(card_count);
    CardMask blocked = state.board.mask;
    for (int i = 0; i < card_count; ++i) {
      const CardId card = SelectAvailableCard(blocked, choices[cursor++]);
      cards.push_back(card);
      blocked |= CardBit(card);
    }

    const ExactGameState parent = state;
    state = ApplyChance(state, cards);
    CheckChanceTransition(parent, state, cards, total_chips);
  }
}

TEST_CASE("betting abstraction uses exact state and street overrides") {
  SolverConfig config;
  config.bet_sizes = {0.5};
  config.flop_bet_sizes = {1.0};
  const ActionMenu override_menu =
      BettingAbstraction(config).actions_for_betting_node(FlopState().betting);
  CHECK(HasAction(override_menu, ActionKind::kBet, 4));
  CHECK_FALSE(HasAction(override_menu, ActionKind::kBet, 2));

  config.flop_bet_sizes.clear();
  BettingState deep;
  deep.stack = {200, 200};
  deep.committed = {50, 50};
  deep.street = StreetKind::kFlop;
  deep.player_to_act = 0;
  deep.folded_player = -1;
  const ActionMenu exact_menu =
      BettingAbstraction(config).actions_for_betting_node(deep);
  CHECK(HasAction(exact_menu, ActionKind::kBet, 50));
  CHECK(HasAction(exact_menu, ActionKind::kAllIn, 200));
}

TEST_CASE("representative betting sequences finish the correct round") {
  struct CompletionCase {
    const char* name;
    ExactGameState initial;
    std::vector<GameAction> actions;
    bool terminal;
  };
  const CompletionCase cases[] = {
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
    ExactGameState result = test.initial;
    for (const GameAction& action : test.actions) {
      result.betting = ApplyAction(result.betting, action);
    }
    CHECK(IsBettingRoundOver(result.betting));
    CHECK(result.betting.player_to_act == -1);
    CHECK(IsTerminal(result.betting, result.board) == test.terminal);
  }
}

TEST_CASE("short all-ins and board runout preserve terminal semantics") {
  ExactGameState short_call = PreflopState();
  short_call.betting.stack = {3, 12};
  short_call.betting.committed = {1, 8};
  short_call = ApplySequence(short_call, {{ActionKind::kCall}});
  CHECK(IsBettingRoundOver(short_call.betting));
  CHECK(short_call.betting.player_to_act == -1);
  CHECK(!IsTerminal(short_call.betting, short_call.board));
  CHECK(short_call.betting.stack[0] == 0);
  CHECK(short_call.betting.committed[0] < short_call.betting.committed[1]);

  ExactGameState state = ApplySequence(
      PreflopState(), {{ActionKind::kCall}, {ActionKind::kCheck}});
  state = ApplyChance(state, {Card(2, SuitKind::kHearts),
                              Card(7, SuitKind::kDiamonds),
                              Card(12, SuitKind::kClubs)});
  state = ApplySequence(state,
                        {{ActionKind::kAllIn}, {ActionKind::kCall}});
  CHECK(IsBettingRoundOver(state.betting));
  state = ApplyChance(state, {Card(9, SuitKind::kSpades)});
  CHECK(state.betting.street == StreetKind::kTurn);
  CHECK(state.betting.player_to_act == -1);
  state = ApplyChance(state, {Card(3, SuitKind::kHearts)});
  CHECK(IsTerminal(state.betting, state.board));
}

TEST_CASE("invalid transitions are rejected and irrelevant amounts ignored") {
  ExactGameState preflop = ApplySequence(
      PreflopState(), {{ActionKind::kCall}, {ActionKind::kCheck}});
  CHECK_THROWS_AS(ApplyChance(preflop, {Card(14, SuitKind::kSpades)}),
                  std::invalid_argument);

  ExactGameState flop = ApplyChance(
      preflop, {Card(2, SuitKind::kHearts), Card(7, SuitKind::kDiamonds),
                Card(12, SuitKind::kClubs)});
  flop = ApplySequence(flop,
                       {{ActionKind::kCheck}, {ActionKind::kCheck}});
  CHECK_THROWS_AS(ApplyChance(flop, {Card(9, SuitKind::kSpades),
                                     Card(3, SuitKind::kHearts)}),
                  std::invalid_argument);
  ExactGameState turn =
      ApplyChance(flop, {Card(9, SuitKind::kSpades)});
  turn = ApplySequence(turn,
                       {{ActionKind::kCheck}, {ActionKind::kCheck}});
  CHECK_THROWS_AS(ApplyChance(turn, {}), std::invalid_argument);

  const BettingState root = PreflopState().betting;
  CHECK_THROWS(ApplyAction(root, {ActionKind::kCheck}));
  CHECK_THROWS(ApplyAction(root, {ActionKind::kRaise, 1}));
  CHECK_THROWS(ApplyAction(root,
                           {ActionKind::kRaise, root.stack[0]}));
  const BettingState flop_betting = FlopState().betting;
  CHECK_THROWS(ApplyAction(flop_betting, {ActionKind::kCall}));
  CHECK_THROWS(ApplyAction(
      flop_betting, {ActionKind::kBet, flop_betting.stack[1]}));
  const BettingState folded =
      ApplySequence(PreflopState(), {{ActionKind::kFold}}).betting;
  CHECK_THROWS(ApplyAction(folded, {ActionKind::kCall}));
  BettingState broke = flop_betting;
  broke.stack[1] = 0;
  CHECK_THROWS(ApplyAction(broke, {ActionKind::kCheck}));

  CHECK(ApplyAction(root, {ActionKind::kCall}) ==
        ApplyAction(root, {ActionKind::kCall, 999}));
  CHECK(ApplyAction(root, {ActionKind::kFold}) ==
        ApplyAction(root, {ActionKind::kFold, 999}));
  CHECK(ApplyAction(flop_betting, {ActionKind::kCheck}) ==
        ApplyAction(flop_betting, {ActionKind::kCheck, 999}));
  CHECK(ApplyAction(flop_betting, {ActionKind::kAllIn}) ==
        ApplyAction(flop_betting, {ActionKind::kAllIn, 999}));
}

TEST_CASE("terminal utility matches an independent zero-sum oracle") {
  CheckUtility(ApplySequence(PreflopState(), {{ActionKind::kFold}}),
               Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts),
               Combo(12, SuitKind::kClubs, 11, SuitKind::kClubs), -1.0);

  CheckUtility(RiverShowdown({Card(14, SuitKind::kHearts),
                              Card(13, SuitKind::kHearts),
                              Card(12, SuitKind::kHearts),
                              Card(11, SuitKind::kHearts),
                              Card(2, SuitKind::kClubs)}),
               Combo(10, SuitKind::kHearts, 4, SuitKind::kClubs),
               Combo(9, SuitKind::kHearts, 5, SuitKind::kClubs), 10.0);

  CheckUtility(RiverShowdown({Card(2, SuitKind::kHearts),
                              Card(3, SuitKind::kDiamonds),
                              Card(4, SuitKind::kClubs),
                              Card(5, SuitKind::kSpades),
                              Card(6, SuitKind::kHearts)}),
               Combo(14, SuitKind::kClubs, 13, SuitKind::kDiamonds),
               Combo(12, SuitKind::kClubs, 11, SuitKind::kDiamonds), 0.0);
}

TEST_CASE("reachable game states preserve all core invariants") {
  const bool passed = rc::check("reachable game semantics", [] {
    const int starting_stack = *rc::gen::inRange(4, 33).as("stack");
    const auto size_points =
        *rc::gen::resize(5, rc::gen::arbitrary<std::vector<uint8_t>>())
             .as("bet sizes");
    const auto choices =
        *rc::gen::arbitrary<std::array<uint16_t, 64>>().as("choices");
    ReplayAndValidate(starting_stack, size_points, choices);
  });
  CHECK(passed);
}

}  // namespace
}  // namespace poker
