#include "tests/rules_test_support.h"

#include "doctest/doctest.h"
#include "src/poker.h"

#include <algorithm>
#include <array>
#include <bit>
#include <random>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

constexpr BettingRules kRules{2};

using S = SuitKind;

CardId C(int rank, S suit) { return MakeCardId(rank, suit); }

ComboId H(int first_rank,
          S first_suit,
          int second_rank,
          S second_suit) {
  return CardsToComboId(C(first_rank, first_suit),
                        C(second_rank, second_suit));
}

std::array<CardId, 3> Flop() {
  return {
      MakeCardId(2, SuitKind::kHearts),
      MakeCardId(7, SuitKind::kDiamonds),
      MakeCardId(12, SuitKind::kClubs),
  };
}

SolverActions ActionsFor(const BettingState& state,
                         absl::Span<const double> sizes) {
  SolverConfig config;
  config.bet_sizes[static_cast<size_t>(state.street)].assign(
      sizes.begin(), sizes.end());
  return GetSolverActions(config, state);
}

ExactPublicState ClosedState(StreetKind street) {
  ExactPublicState state;
  state.betting.stack = {10, 10};
  state.betting.total_committed = {10, 10};
  state.betting.street_committed = {0, 0};
  state.betting.last_full_raise = 2;
  state.betting.street = street;
  state.betting.player_to_act = -1;
  state.betting.pending_action_mask = 0;

  if (street == StreetKind::kPreflop) {
    return state;
  }
  state.board.deal_flop(Flop());
  if (street == StreetKind::kFlop) {
    return state;
  }
  state.board.deal_turn(C(9, S::kSpades));
  if (street == StreetKind::kTurn) {
    return state;
  }
  state.board.deal_river(C(3, S::kHearts));
  return state;
}

ExactPublicState Showdown(std::array<CardId, kMaxBoardCards> cards) {
  ExactPublicState state = ClosedState(StreetKind::kRiver);
  state.board = BoardRunout::Preflop();
  state.board.deal_flop(absl::Span<const CardId>(cards.data(), 3));
  state.board.deal_turn(cards[3]);
  state.board.deal_river(cards[4]);
  return state;
}

enum class StatePhase {
  kDecision,
  kChance,
  kTerminal,
};

void CheckExactState(const char* label,
                     const ExactPublicState& actual,
                     std::array<Chips, kPlayerCount> stack,
                     std::array<Chips, kPlayerCount> total,
                     std::array<Chips, kPlayerCount> street_committed,
                     Chips last_full_raise,
                     StreetKind street,
                     int player_to_act,
                     uint8_t pending_action_mask,
                     const BoardRunout& expected_board,
                     StatePhase expected_phase) {
  CAPTURE(label);
  BettingState expected_betting;
  expected_betting.stack = stack;
  expected_betting.total_committed = total;
  expected_betting.street_committed = street_committed;
  expected_betting.last_full_raise = last_full_raise;
  expected_betting.street = street;
  expected_betting.player_to_act = static_cast<int8_t>(player_to_act);
  expected_betting.pending_action_mask = pending_action_mask;
  CHECK(actual.betting == expected_betting);
  CHECK(Pot(actual.betting) == Pot(expected_betting));
  CHECK(actual.board == expected_board);

  if (expected_phase == StatePhase::kTerminal) {
    CHECK(IsTerminal(actual));
    return;
  }
  CHECK_FALSE(IsTerminal(actual));
  if (expected_phase == StatePhase::kChance) {
    CHECK(IsBettingRoundOver(actual.betting));
    CHECK(actual.betting.player_to_act == -1);
    return;
  }
  CHECK_FALSE(IsBettingRoundOver(actual.betting));
  CHECK(IsPlayer(actual.betting.player_to_act));
}

void CheckGeneralInvariants(
    const ExactPublicState& state,
    const std::array<Chips, kPlayerCount>& initial_chips) {
  CHECK(IsValidBettingState(state.betting));
  for (size_t player = 0; player < kPlayerCount; ++player) {
    CHECK(state.betting.stack[player] >= 0);
    CHECK(state.betting.total_committed[player] >= 0);
    CHECK(state.betting.street_committed[player] >= 0);
    CHECK(state.betting.street_committed[player] <=
          state.betting.total_committed[player]);
    CHECK(state.betting.stack[player] +
              state.betting.total_committed[player] ==
          initial_chips[player]);
  }
  CHECK(Pot(state.betting) == state.betting.total_committed[0] +
                                  state.betting.total_committed[1]);

  CardMask seen = 0;
  for (CardId card : state.board.cards()) {
    CHECK((seen & CardBit(card)) == 0);
    seen |= CardBit(card);
  }
  CHECK(seen == state.board.mask());
  CHECK(std::popcount(state.board.mask()) == state.board.count());
  CHECK(BoardCardsForStreet(state.betting.street) == state.board.count());

  if (IsBettingRoundOver(state.betting)) {
    CHECK(state.betting.player_to_act == -1);
    if (state.betting.folded_player < 0) {
      CHECK(state.betting.street_committed[0] ==
            state.betting.street_committed[1]);
    }
  } else {
    CHECK(IsPlayer(state.betting.player_to_act));
  }
  if (IsTerminal(state)) {
    CHECK(state.betting.player_to_act == -1);
  }
}

void CheckGeneratedRollout(uint32_t seed) {
  CAPTURE(seed);
  ExactPublicState state =
      MakeInitialState(kRules, {20, 20}, {1, 2});
  const std::array<Chips, kPlayerCount> initial_chips = {20, 20};
  const std::array<double, 3> sizes = {1.0, 0.25, 0.5};
  const std::array<double, 3> sorted_sizes = {0.25, 0.5, 1.0};
  std::array<CardId, kDeckCardCount> deck = {};
  for (int id = 0; id < kDeckCardCount; ++id) {
    deck[static_cast<size_t>(id)] = static_cast<CardId>(id);
  }
  std::mt19937 rng(seed);

  CheckGeneralInvariants(state, initial_chips);
  for (int step = 0; step < 64 && !IsTerminal(state); ++step) {
    if (IsPlayer(state.betting.player_to_act)) {
      const SolverActions menu = ActionsFor(state.betting, sizes);
      const SolverActions canonical =
          ActionsFor(state.betting, sorted_sizes);
      REQUIRE(menu.size() > 0);
      REQUIRE(menu.size() == canonical.size());
      for (size_t i = 0; i < menu.size(); ++i) {
        CHECK(menu[i] == canonical[i]);
        ExactPublicState child = state;
        child.betting = ApplyAction(state.betting, menu[i]);
        CheckGeneralInvariants(child, initial_chips);
        CHECK(child.board == state.board);
      }
      const size_t index = rng() % menu.size();
      state.betting = ApplyAction(state.betting, menu[index]);
      CheckGeneralInvariants(state, initial_chips);
      continue;
    }

    std::shuffle(deck.begin(), deck.end(), rng);
    std::vector<CardId> cards;
    for (CardId card : deck) {
      if (!state.board.contains(card)) {
        cards.push_back(card);
      }
      if (cards.size() ==
          static_cast<size_t>(CardsForNextStreet(state.betting.street))) {
        break;
      }
    }
    const BettingState before = state.betting;
    const CardMask board_before = state.board.mask();
    state = ApplyChance(state, cards, kRules);
    CheckGeneralInvariants(state, initial_chips);
    CHECK(state.betting.stack == before.stack);
    CHECK(state.betting.total_committed == before.total_committed);
    CHECK((state.board.mask() & board_before) == board_before);
  }
  CHECK(IsTerminal(state));
}

TEST_CASE("generated transitions preserve exact-state invariants") {
  for (uint32_t seed = 0; seed < 64; ++seed) {
    CheckGeneratedRollout(seed);
  }
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

bool HasAction(const SolverActions& menu, GameAction expected) {
  for (const GameAction& action : menu) {
    if (action == expected) {
      return true;
    }
  }
  return false;
}

void CheckMenu(const BettingState& state,
               absl::Span<const double> sizes) {
  const SolverActions menu = ActionsFor(state, sizes);
  for (const GameAction& action : menu) {
    CHECK_NOTHROW(ApplyAction(state, action));
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
  CHECK_FALSE(IsTerminal(state));
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
  const SolverActions menu = ActionsFor(preflop, subminimum_size);
  CHECK_FALSE(HasAction(menu, {ActionKind::kRaise, 3}));
  CheckMenu(preflop, subminimum_size);
}

TEST_CASE("big blind retains the raise option after a limp") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  const std::array<double, 1> sizes = {0.5};
  const SolverActions menu = ActionsFor(state.betting, sizes);

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
    const SolverActions menu = ActionsFor(state, sizes);
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
    const SolverActions menu = ActionsFor(state, sizes);
    CHECK(HasAction(menu, {ActionKind::kAllIn, 20}));
    for (const GameAction& action : menu) {
      CHECK(action.target_street_commitment <= 20);
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
  const SolverActions settled_menu = ActionsFor(settled, sizes);
  CHECK(settled_menu.size() == 1);
  CHECK(HasAction(settled_menu, {ActionKind::kCheck, 0}));

  const BettingState facing_bet =
      State({20, 0}, {0, 10}, {0, 10}, 0, 2);
  const SolverActions call_menu = ActionsFor(facing_bet, sizes);
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
  CHECK_FALSE(IsTerminal(state));

  const std::array<CardId, 1> turn = {
      MakeCardId(9, SuitKind::kSpades),
  };
  state = ApplyChance(state, turn, kRules);
  CHECK(state.betting.player_to_act == -1);
  CHECK_FALSE(IsTerminal(state));

  const std::array<CardId, 1> river = {
      MakeCardId(3, SuitKind::kHearts),
  };
  state = ApplyChance(state, river, kRules);
  CHECK(state.betting.player_to_act == -1);
  CHECK(IsTerminal(state));
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
  CHECK(IsTerminal(state));
  CHECK(state.betting.total_committed[0] !=
        state.betting.total_committed[1]);
  CHECK(state.betting.folded_player == 0);
  CHECK(state.betting.player_to_act == -1);
}

TEST_CASE("folds are terminal on every street") {
  const std::array<StreetKind, 4> streets = {
      StreetKind::kPreflop,
      StreetKind::kFlop,
      StreetKind::kTurn,
      StreetKind::kRiver,
  };
  const ComboId player0 = H(14, S::kSpades, 13, S::kSpades);
  const ComboId player1 = H(12, S::kClubs, 11, S::kClubs);

  for (StreetKind street : streets) {
    ExactPublicState state = ClosedState(street);
    state.betting.folded_player = 0;
    CHECK(IsTerminal(state));
    CHECK(TerminalUtility(state, player0, player1) ==
          doctest::Approx(-10.0));
  }
}

TEST_CASE("terminal utility rejects nonterminal states") {
  const ComboId player0 = H(14, S::kSpades, 13, S::kSpades);
  const ComboId player1 = H(12, S::kClubs, 11, S::kClubs);

  for (StreetKind street : {StreetKind::kPreflop, StreetKind::kFlop,
                            StreetKind::kTurn}) {
    const ExactPublicState state = ClosedState(street);
    CHECK_FALSE(IsTerminal(state));
    CHECK_THROWS_AS(TerminalUtility(state, player0, player1),
                    std::invalid_argument);
  }

  ExactPublicState river = ClosedState(StreetKind::kRiver);
  river.betting.player_to_act = 0;
  river.betting.pending_action_mask = kAllPlayersMask;
  CHECK_FALSE(IsTerminal(river));
  CHECK_THROWS_AS(TerminalUtility(river, player0, player1),
                  std::invalid_argument);
}

TEST_CASE("river terminal utility handles win, loss, and tie") {
  const ComboId player0 = H(14, S::kHearts, 13, S::kHearts);
  const ComboId player1 = H(9, S::kHearts, 8, S::kHearts);
  const ExactPublicState win = Showdown({
      C(10, S::kHearts),
      C(11, S::kHearts),
      C(12, S::kHearts),
      C(2, S::kClubs),
      C(3, S::kDiamonds),
  });
  REQUIRE(win.betting.total_committed[0] ==
          win.betting.total_committed[1]);
  const double win_utility = TerminalUtility(win, player0, player1);
  const double loss_utility = TerminalUtility(win, player1, player0);
  CHECK(win_utility == doctest::Approx(10.0));
  CHECK(loss_utility == doctest::Approx(-10.0));
  CHECK(win_utility + loss_utility == doctest::Approx(0.0));

  const ExactPublicState tie = Showdown({
      C(2, S::kHearts),
      C(3, S::kDiamonds),
      C(4, S::kClubs),
      C(5, S::kSpades),
      C(6, S::kHearts),
  });
  REQUIRE(tie.betting.total_committed[0] ==
          tie.betting.total_committed[1]);
  CHECK(TerminalUtility(
            tie, H(14, S::kClubs, 13, S::kDiamonds),
            H(12, S::kClubs, 11, S::kDiamonds)) == doctest::Approx(0.0));
}

TEST_CASE("a complete normal hand preserves exact state") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  BoardRunout board = BoardRunout::Preflop();
  CheckExactState("blinds", state, {19, 18}, {1, 2}, {1, 2}, 2,
                  StreetKind::kPreflop, 0, kAllPlayersMask, board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  CheckExactState("small blind calls", state, {18, 18}, {2, 2},
                  {2, 2}, 2, StreetKind::kPreflop, 1, PlayerBit(1),
                  board, StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CheckExactState("big blind checks", state, {18, 18}, {2, 2},
                  {2, 2}, 2, StreetKind::kPreflop, -1, 0, board,
                  StatePhase::kChance);

  state = ApplyChance(state, Flop(), kRules);
  board.deal_flop(Flop());
  CheckExactState("flop", state, {18, 18}, {2, 2}, {0, 0}, 2,
                  StreetKind::kFlop, 1, kAllPlayersMask, board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CheckExactState("flop check", state, {18, 18}, {2, 2}, {0, 0}, 2,
                  StreetKind::kFlop, 0, PlayerBit(0), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kBet, 2});
  CheckExactState("flop bet", state, {16, 18}, {4, 2}, {2, 0}, 2,
                  StreetKind::kFlop, 1, PlayerBit(1), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  CheckExactState("flop call", state, {16, 16}, {4, 4}, {2, 2}, 2,
                  StreetKind::kFlop, -1, 0, board,
                  StatePhase::kChance);

  const std::array<CardId, 1> turn = {C(9, S::kSpades)};
  state = ApplyChance(state, turn, kRules);
  board.deal_turn(turn[0]);
  CheckExactState("turn", state, {16, 16}, {4, 4}, {0, 0}, 2,
                  StreetKind::kTurn, 1, kAllPlayersMask, board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CheckExactState("turn first check", state, {16, 16}, {4, 4},
                  {0, 0}, 2, StreetKind::kTurn, 0, PlayerBit(0), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  CheckExactState("turn second check", state, {16, 16}, {4, 4},
                  {0, 0}, 2, StreetKind::kTurn, -1, 0, board,
                  StatePhase::kChance);

  const std::array<CardId, 1> river = {C(3, S::kHearts)};
  state = ApplyChance(state, river, kRules);
  board.deal_river(river[0]);
  CheckExactState("river", state, {16, 16}, {4, 4}, {0, 0}, 2,
                  StreetKind::kRiver, 1, kAllPlayersMask, board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kBet, 2});
  CheckExactState("river bet", state, {16, 14}, {4, 6}, {0, 2}, 2,
                  StreetKind::kRiver, 0, PlayerBit(0), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  CheckExactState("river call", state, {14, 14}, {6, 6}, {2, 2}, 2,
                  StreetKind::kRiver, -1, 0, board,
                  StatePhase::kTerminal);

  const ComboId aces = H(14, S::kClubs, 14, S::kDiamonds);
  const ComboId kings = H(13, S::kClubs, 13, S::kDiamonds);
  CHECK(TerminalUtility(state, aces, kings) == doctest::Approx(6.0));
}

TEST_CASE("full raises update the minimum re-raise increment") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  const BoardRunout board = BoardRunout::Preflop();

  state.betting = ApplyAction(state.betting, {ActionKind::kRaise, 4});
  CheckExactState("minimum raise", state, {16, 18}, {4, 2}, {4, 2}, 2,
                  StreetKind::kPreflop, 1, PlayerBit(1), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kRaise, 8});
  CheckExactState("full re-raise", state, {16, 12}, {4, 8}, {4, 8}, 4,
                  StreetKind::kPreflop, 0, PlayerBit(0), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 8});
  CheckExactState("call re-raise", state, {12, 12}, {8, 8}, {8, 8}, 4,
                  StreetKind::kPreflop, -1, 0, board,
                  StatePhase::kChance);
}

TEST_CASE("a short all-in raise preserves the full-raise increment") {
  ExactPublicState state = test::InitialHeadsUpState(20, 5, 1, 2);
  BoardRunout board = BoardRunout::Preflop();

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  state = ApplyChance(state, Flop(), kRules);
  board.deal_flop(Flop());
  CheckExactState("short stack reaches flop", state, {18, 3}, {2, 2},
                  {0, 0}, 2, StreetKind::kFlop, 1, kAllPlayersMask,
                  board, StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  state.betting = ApplyAction(state.betting, {ActionKind::kBet, 2});
  CheckExactState("player zero bets", state, {16, 3}, {4, 2}, {2, 0}, 2,
                  StreetKind::kFlop, 1, PlayerBit(1), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kAllIn, 3});
  CheckExactState("subminimum all-in raise", state, {16, 0}, {4, 5},
                  {2, 3}, 2, StreetKind::kFlop, 0, PlayerBit(0), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 3});
  CheckExactState("short raise called", state, {15, 0}, {5, 5}, {3, 3},
                  2, StreetKind::kFlop, -1, 0, board,
                  StatePhase::kChance);

  const std::array<CardId, 1> turn = {C(9, S::kSpades)};
  state = ApplyChance(state, turn, kRules);
  board.deal_turn(turn[0]);
  CheckExactState("automatic turn", state, {15, 0}, {5, 5}, {0, 0}, 2,
                  StreetKind::kTurn, -1, kAllPlayersMask, board,
                  StatePhase::kChance);

  const std::array<CardId, 1> river = {C(3, S::kHearts)};
  state = ApplyChance(state, river, kRules);
  board.deal_river(river[0]);
  CheckExactState("automatic river", state, {15, 0}, {5, 5}, {0, 0}, 2,
                  StreetKind::kRiver, -1, kAllPlayersMask, board,
                  StatePhase::kTerminal);
}

TEST_CASE("effective stacks leave unmatched chips uncommitted") {
  ExactPublicState state = test::InitialHeadsUpState(100, 20, 1, 2);
  BoardRunout board = BoardRunout::Preflop();
  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 2});
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});
  state = ApplyChance(state, Flop(), kRules);
  board.deal_flop(Flop());
  state.betting = ApplyAction(state.betting, {ActionKind::kCheck});

  const std::array<double, 2> sizes = {1.0, 10.0};
  const SolverActions menu = ActionsFor(state.betting, sizes);
  for (const GameAction& action : menu) {
    CHECK(action.target_street_commitment <= 18);
  }

  state.betting = ApplyAction(state.betting, {ActionKind::kAllIn, 18});
  CheckExactState("effective all-in", state, {80, 18}, {20, 2}, {18, 0},
                  18, StreetKind::kFlop, 1, PlayerBit(1), board,
                  StatePhase::kDecision);

  state.betting = ApplyAction(state.betting, {ActionKind::kCall, 18});
  CheckExactState("effective all-in called", state, {80, 0}, {20, 20},
                  {18, 18}, 18, StreetKind::kFlop, -1, 0, board,
                  StatePhase::kChance);
}

TEST_CASE("solver action count follows configured bet sizes") {
  const BettingState state =
      State({1000, 1000}, {100, 100}, {0, 0}, 0, 2);
  const std::array<double, 9> sizes = {
      0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18};

  const SolverActions actions = ActionsFor(state, sizes);

  CHECK(actions.size() == 11);
  for (const GameAction& action : actions) {
    CHECK_NOTHROW(ApplyAction(state, action));
  }
}

}  // namespace
}  // namespace poker
