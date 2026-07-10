#include "src/betting_abstraction.h"
#include "src/card_utils.h"
#include "src/combo.h"
#include "src/game_rules.h"

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <bit>
#include <initializer_list>
#include <random>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

using S = SuitKind;

CardId C(int rank, S suit) { return MakeCardId(rank, suit); }
ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(C(r0, s0), C(r1, s1));
}

BoardRunout RiverRunout(std::array<CardId, 5> cards) {
  BoardRunout runout = BoardRunout::Preflop();
  runout.deal_flop(absl::Span<const CardId>(cards.data(), 3));
  runout.deal_turn(cards[3]);
  runout.deal_river(cards[4]);
  return runout;
}

ExactPublicState Root() {
  ExactPublicState state;
  state.betting.stack = {19, 18};
  state.betting.total_committed = {1, 2};
  state.betting.street_committed = {1, 2};
  state.betting.last_full_raise = 2;
  return state;
}

constexpr BettingRules kRules{2};

int ChipsInPlay(const ExactPublicState& state) {
  return Pot(state.betting) + state.betting.stack[0] + state.betting.stack[1];
}

void CheckState(const ExactPublicState& state, int total) {
  CHECK(IsValidBettingState(state.betting));
  CHECK(ChipsInPlay(state) == total);
  CHECK(Pot(state.betting) ==
        state.betting.total_committed[0] + state.betting.total_committed[1]);
  CHECK(BoardCardsForStreet(state.betting.street) == state.board.count());
  CHECK(std::popcount(state.board.mask()) == state.board.count());
  CHECK(IsBettingRoundOver(state.betting) ==
        !IsPlayer(state.betting.player_to_act));
}

void Rollout(uint32_t seed) {
  ExactPublicState state = Root();
  const int total = ChipsInPlay(state);
  const std::array<double, 3> sizes = {1.0, 0.25, 0.5};
  const std::array<double, 3> sorted_sizes = {0.25, 0.5, 1.0};
  std::array<CardId, kDeckCardCount> deck = {};
  for (int i = 0; i < kDeckCardCount; ++i) deck[i] = static_cast<CardId>(i);
  std::mt19937 rng(seed);

  for (int step = 0; step < 64 && !IsTerminal(state.betting, state.board);
       ++step) {
    CheckState(state, total);
    if (IsPlayer(state.betting.player_to_act)) {
      const ActionMenu menu = LegalActions(state.betting, sizes);
      const ActionMenu canonical = LegalActions(state.betting, sorted_sizes);
      REQUIRE(menu.count > 0);
      REQUIRE(menu.count == canonical.count);
      for (uint8_t i = 0; i < menu.count; ++i) {
        CHECK(menu.actions[i] == canonical.actions[i]);
        ExactPublicState child = state;
        child.betting = ApplyAction(state.betting, menu.actions[i]);
        CheckState(child, total);
        CHECK(child.board.mask() == state.board.mask());
      }
      state.betting = ApplyAction(
          state.betting, menu.actions[rng() % static_cast<uint32_t>(menu.count)]);
      continue;
    }

    std::shuffle(deck.begin(), deck.end(), rng);
    std::vector<CardId> cards;
    for (CardId card : deck) {
      if (!state.board.contains(card)) cards.push_back(card);
      if (cards.size() ==
          static_cast<size_t>(CardsForNextStreet(state.betting.street)))
        break;
    }
    const BettingState before = state.betting;
    const CardMask board_before = state.board.mask();
    state = ApplyChance(state, cards, kRules);
    CheckState(state, total);
    CHECK(state.betting.stack == before.stack);
    CHECK(state.betting.total_committed == before.total_committed);
    CHECK((state.board.mask() & board_before) == board_before);
  }
  CheckState(state, total);
  CHECK(IsTerminal(state.betting, state.board));
}

TEST_CASE("legal rollouts preserve game-state invariants") {
  for (uint32_t seed = 0; seed < 64; ++seed) Rollout(seed);
}

TEST_CASE("boundary actions, chance transitions, and sizing are enforced") {
  ExactPublicState state = Root();
  CHECK_THROWS(ApplyAction(state.betting, {ActionKind::kCheck}));
  CHECK_THROWS(ApplyAction(state.betting, {ActionKind::kRaise, 1}));
  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CHECK_THROWS_AS(ApplyChance(state, {C(14, S::kSpades)}, kRules),
                  std::invalid_argument);

  ExactPublicState short_call = Root();
  short_call.betting.stack = {3, 12};
  short_call.betting.total_committed = {1, 8};
  short_call.betting.street_committed = {1, 8};
  short_call.betting = ApplyAction(short_call.betting,
                                   {ActionKind::kCall, 4});
  CHECK(IsBettingRoundOver(short_call.betting));
  CHECK(short_call.betting.stack[0] == 0);
  CHECK(short_call.betting.total_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(short_call.betting.street_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(short_call.betting.stack[1] == 16);

  SolverConfig config;
  config.bet_sizes = {0.5};
  config.flop_bet_sizes = {1.0};
  BettingState flop;
  flop.stack = {98, 98};
  flop.total_committed = {2, 2};
  flop.last_full_raise = 2;
  flop.street = StreetKind::kFlop;
  flop.player_to_act = 1;
  flop.folded_player = -1;
  const ActionMenu menu = BettingAbstraction(config).actions_for_betting_node(flop);
  bool bet_four = false;
  bool bet_two = false;
  bool all_in = false;
  for (uint8_t i = 0; i < menu.count; ++i) {
    bet_four |= menu.actions[i] == GameAction{ActionKind::kBet, 4};
    bet_two |= menu.actions[i] == GameAction{ActionKind::kBet, 2};
    all_in |= menu.actions[i] == GameAction{ActionKind::kAllIn, 98};
  }
  CHECK(bet_four);
  CHECK_FALSE(bet_two);
  CHECK(all_in);
}

TEST_CASE("terminal utility handles fold, win, and tie") {
  ExactPublicState folded = Root();
  folded.betting = ApplyAction(folded.betting, {ActionKind::kFold});
  CHECK(GetUtility(folded, H(14, S::kHearts, 13, S::kHearts),
                   H(12, S::kClubs, 11, S::kClubs)) == doctest::Approx(-1.0));

  ExactPublicState showdown;
  showdown.betting.street = StreetKind::kRiver;
  showdown.betting.player_to_act = -1;
  showdown.betting.pending_action_mask = 0;
  showdown.betting.total_committed = {10, 10};
  showdown.betting.street_committed = {10, 10};
  showdown.betting.last_full_raise = 2;
  showdown.board = RiverRunout({
      C(10, S::kHearts),
      C(11, S::kHearts),
      C(12, S::kHearts),
      C(2, S::kClubs),
      C(3, S::kDiamonds),
  });
  CHECK(GetUtility(showdown, H(14, S::kHearts, 13, S::kHearts),
                   H(9, S::kHearts, 8, S::kHearts)) == doctest::Approx(10.0));

  showdown.board = RiverRunout({
      C(2, S::kHearts),
      C(3, S::kDiamonds),
      C(4, S::kClubs),
      C(5, S::kSpades),
      C(6, S::kHearts),
  });
  CHECK(GetUtility(showdown, H(14, S::kClubs, 13, S::kDiamonds),
                   H(12, S::kClubs, 11, S::kDiamonds)) == doctest::Approx(0.0));
}

}  // namespace
}  // namespace poker
