#include "src/betting_abstraction.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "src/card_utils.h"
#include "src/combo.h"
#include "src/game_rules.h"
#include "src/hand_evaluator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <ostream>
#include <vector>

namespace poker {
namespace {

struct ReachableScenario {
  int starting_stack = 20;
  std::vector<uint16_t> choices;
};

std::ostream& operator<<(std::ostream& out,
                         const ReachableScenario& scenario) {
  out << "stack=" << scenario.starting_stack << " choices=[";
  for (size_t i = 0; i < scenario.choices.size(); ++i) {
    out << (i == 0 ? "" : ",") << scenario.choices[i];
  }
  return out << "]";
}

auto ReachableScenarios() {
  return rc::gen::build<ReachableScenario>(
      rc::gen::set(&ReachableScenario::starting_stack,
                   rc::gen::inRange(4, 33)),
      rc::gen::set(&ReachableScenario::choices,
                   rc::gen::resize(
                       64, rc::gen::arbitrary<std::vector<uint16_t>>())))
      .as("scenario");
}

SolverConfig TestConfig() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {0.5, 1.0};
  return config;
}

ExactGameState InitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;

  ExactGameState state;
  state.betting.stack[0] =
      std::max(0, config.starting_stack_size - small_blind);
  state.betting.stack[1] =
      std::max(0, config.starting_stack_size - big_blind);
  state.betting.committed = {small_blind, big_blind};
  return state;
}

bool Terminal(const ExactGameState& state) {
  return IsTerminal(state.betting, state.board);
}

ExactGameState ApplyStateAction(ExactGameState state,
                                const GameAction& action) {
  state.betting = ApplyAction(state.betting, action);
  return state;
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
                           absl::Span<const CardId> cards,
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

void ReplayAndValidate(const ReachableScenario& scenario) {
  SolverConfig config = TestConfig();
  config.starting_stack_size = scenario.starting_stack;
  const BettingAbstraction betting(config);
  ExactGameState state = InitialState(config);
  const int total_chips = TotalChips(state);
  size_t cursor = 0;

  while (true) {
    CheckState(state, total_chips);
    if (Terminal(state)) {
      return;
    }

    const int player = state.betting.player_to_act;
    if (IsPlayer(player)) {
      const ActionMenu menu =
          betting.actions_for_betting_node(state.betting);
      RC_ASSERT(menu.count > 0);
      for (uint8_t i = 0; i < menu.count; ++i) {
        ExactGameState child = state;
        child.betting = ApplyAction(state.betting, menu.actions[i]);
        CheckActionTransition(state, child, player, total_chips);
      }
      if (cursor == scenario.choices.size()) {
        return;
      }
      const size_t action = scenario.choices[cursor++] % menu.count;
      state.betting = ApplyAction(state.betting, menu.actions[action]);
      continue;
    }

    const int card_count = CardsForNextStreet(state.betting.street);
    if (cursor + static_cast<size_t>(card_count) > scenario.choices.size()) {
      return;
    }

    std::vector<CardId> cards;
    cards.reserve(static_cast<size_t>(card_count));
    CardMask blocked = state.board.mask;
    for (int i = 0; i < card_count; ++i) {
      const CardId card =
          SelectAvailableCard(blocked, scenario.choices[cursor++]);
      cards.push_back(card);
      blocked |= CardBit(card);
    }

    const ExactGameState parent = state;
    state = ApplyChance(state, cards);
    CheckChanceTransition(parent, state, cards, total_chips);
  }
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

TEST_CASE("reachable states preserve poker invariants") {
  const bool passed = rc::check("reachable game trace", [] {
    const ReachableScenario scenario = *ReachableScenarios();
    ReplayAndValidate(scenario);
  });
  CHECK(passed);
}

TEST_CASE("representative invalid actions are rejected") {
  const ExactGameState root = InitialState(TestConfig());
  CHECK_THROWS(ApplyAction(root.betting, {ActionKind::kCheck}));
  CHECK_THROWS(ApplyAction(root.betting, {ActionKind::kRaise, 1}));
  CHECK_THROWS(ApplyAction(
      root.betting, {ActionKind::kRaise, root.betting.stack[0]}));

  const ExactGameState flop = FlopState();
  CHECK_THROWS(ApplyAction(flop.betting, {ActionKind::kCall}));
  CHECK_THROWS(ApplyAction(
      flop.betting, {ActionKind::kBet, flop.betting.stack[1]}));

  const ExactGameState folded =
      ApplyStateAction(root, {ActionKind::kFold});
  CHECK_THROWS(ApplyAction(folded.betting, {ActionKind::kCall}));

  ExactGameState broke = flop;
  broke.betting.stack[1] = 0;
  CHECK_THROWS(ApplyAction(broke.betting, {ActionKind::kCheck}));
}

TEST_CASE("all-in edge cases complete betting and runout") {
  ExactGameState short_call = InitialState(TestConfig());
  short_call.betting.stack = {3, 12};
  short_call.betting.committed = {1, 8};
  short_call = ApplyStateAction(short_call, {ActionKind::kCall});
  CHECK(IsBettingRoundOver(short_call.betting));
  CHECK(short_call.betting.player_to_act == -1);
  CHECK(!Terminal(short_call));
  CHECK(short_call.betting.stack[0] == 0);
  CHECK(short_call.betting.committed[0] < short_call.betting.committed[1]);

  ExactGameState runout = InitialState(TestConfig());
  runout = ApplyStateAction(runout, {ActionKind::kCall});
  runout = ApplyStateAction(runout, {ActionKind::kCheck});
  runout = ApplyChance(runout, {MakeCardId(2, SuitKind::kHearts),
                                MakeCardId(7, SuitKind::kDiamonds),
                                MakeCardId(12, SuitKind::kClubs)});
  CHECK(runout.betting.street == StreetKind::kFlop);
  CHECK(runout.board.count == 3);
  CHECK(runout.betting.player_to_act == 1);
  runout = ApplyStateAction(runout, {ActionKind::kAllIn});
  runout = ApplyStateAction(runout, {ActionKind::kCall});
  runout = ApplyChance(runout, {MakeCardId(9, SuitKind::kSpades)});
  CHECK(runout.betting.player_to_act == -1);
  runout = ApplyChance(runout, {MakeCardId(3, SuitKind::kHearts)});
  CHECK(Terminal(runout));
  CHECK(runout.betting.player_to_act == -1);
}

TEST_CASE("terminal utility matches an independent oracle") {
  ExactGameState folded = InitialState(TestConfig());
  folded = ApplyStateAction(folded, {ActionKind::kFold});
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
