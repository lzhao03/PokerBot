#include "tests/rules_test_support.h"

#include "doctest/doctest.h"
#include "src/bet_abstraction.h"
#include "src/poker.h"

#include <algorithm>
#include <array>
#include <bit>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker {
namespace {

constexpr BettingRules kRules{2};

using S = Suit;

Card C(int rank, S suit) { return Card(static_cast<Rank>(rank - 2), suit); }

BettingData& B(BettingState& state) {
  return std::visit([](auto& phase) -> BettingData& { return phase.data; },
                    state);
}

const BettingData& B(const BettingState& state) { return Data(state); }
BettingData& B(ExactPublicState& state) { return B(state.betting); }
const BettingData& B(const ExactPublicState& state) {
  return Data(state.betting);
}

ComboId H(int first_rank,
          S first_suit,
          int second_rank,
          S second_suit) {
  return CardsToComboId(C(first_rank, first_suit),
                        C(second_rank, second_suit));
}

BettingState Apply(const BettingState& state, GameAction action) {
  const auto* decision = std::get_if<DecisionState>(&state);
  if (decision == nullptr) {
    throw std::invalid_argument("expected decision state");
  }
  const auto child = ApplyAction(*decision, action);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

ExactPublicState DealChance(const ExactPublicState& state,
                            absl::Span<const Card> cards,
                            const BettingRules& rules) {
  const auto child = TryApplyChance(state, cards, rules);
  if (!child.ok()) {
    throw std::invalid_argument(std::string(child.status().message()));
  }
  return *child;
}

std::array<Card, 3> Flop() {
  return {
      C(2, Suit::Hearts), C(7, Suit::Diamonds),
      C(12, Suit::Clubs),
  };
}

void AddFlop(Board& board, absl::Span<const Card> cards) {
  board = DealCards(board, cards);
}

void AddTurn(Board& board, Card card) {
  board = DealCards(board, std::array<Card, 1>{card});
}

void AddRiver(Board& board, Card card) {
  board = DealCards(board, std::array<Card, 1>{card});
}

std::vector<GameAction> ActionsFor(const BettingState& state,
                                   absl::Span<const double> sizes) {
  const auto* decision = std::get_if<DecisionState>(&state);
  if (decision == nullptr) {
    throw std::invalid_argument("expected decision state");
  }
  BetAbstractionConfig config;
  config.pot_fractions[static_cast<size_t>(decision->data.street)].assign(
      sizes.begin(), sizes.end());
  std::vector<GameAction> actions;
  for (const GameAction& action :
       SelectAbstractActions(config, *decision)) {
    actions.push_back(action);
  }
  return actions;
}

ExactPublicState ClosedState(StreetKind street) {
  ExactPublicState state;
  BettingData data;
  data.stack = {10, 10};
  data.total_committed = {10, 10};
  data.street_committed = {0, 0};
  data.last_full_raise = 2;
  data.street = street;
  data.actions_remaining = 0;
  state.betting = street == StreetKind::River
                      ? BettingState(ShowdownState{data})
                      : BettingState(ChanceState{data});

  if (street == StreetKind::Preflop) {
    return state;
  }
  AddFlop(state.board, Flop());
  if (street == StreetKind::Flop) {
    return state;
  }
  AddTurn(state.board, C(9, S::Spades));
  if (street == StreetKind::Turn) {
    return state;
  }
  AddRiver(state.board, C(3, S::Hearts));
  return state;
}

ExactPublicState Showdown(std::array<Card, kMaxBoardCards> cards) {
  ExactPublicState state = ClosedState(StreetKind::River);
  state.board = Board{};
  AddFlop(state.board, absl::Span<const Card>(cards.data(), 3));
  AddTurn(state.board, cards[3]);
  AddRiver(state.board, cards[4]);
  return state;
}

enum class StatePhase {
  Decision,
  Chance,
  Terminal,
};

void CheckExactState(const char* label,
                     const ExactPublicState& actual,
                     std::array<Chips, kPlayerCount> stack,
                     std::array<Chips, kPlayerCount> total,
                     std::array<Chips, kPlayerCount> street_committed,
                     Chips last_full_raise,
                     StreetKind street,
                     int player_to_act,
                     uint8_t actions_remaining,
                     const Board& expected_board,
                     StatePhase expected_phase) {
  CAPTURE(label);
  BettingData expected_data;
  expected_data.stack = stack;
  expected_data.total_committed = total;
  expected_data.street_committed = street_committed;
  expected_data.last_full_raise = last_full_raise;
  expected_data.street = street;
  expected_data.actions_remaining = actions_remaining;
  BettingState expected_betting;
  if (expected_phase == StatePhase::Decision) {
    expected_betting = DecisionState{
        expected_data, static_cast<Player>(player_to_act)};
  } else if (expected_phase == StatePhase::Chance) {
    expected_betting = ChanceState{expected_data};
  } else {
    expected_betting = ShowdownState{expected_data};
  }
  CHECK(actual.betting == expected_betting);
  CHECK(Pot(B(actual)) == Pot(Data(expected_betting)));
  CHECK(actual.board == expected_board);

  if (expected_phase == StatePhase::Terminal) {
    CHECK(IsTerminal(actual));
    return;
  }
  CHECK_FALSE(IsTerminal(actual));
  if (expected_phase == StatePhase::Chance) {
    CHECK(std::holds_alternative<ChanceState>(actual.betting));
    return;
  }
  CHECK(std::holds_alternative<DecisionState>(actual.betting));
}

void CheckGeneralInvariants(
    const ExactPublicState& state,
    const std::array<Chips, kPlayerCount>& initial_chips) {
  const BettingData& betting = B(state);
  CHECK(IsValidBettingData(betting));
  for (size_t player = 0; player < kPlayerCount; ++player) {
    CHECK(betting.stack[player] >= 0);
    CHECK(betting.total_committed[player] >= 0);
    CHECK(betting.street_committed[player] >= 0);
    CHECK(betting.street_committed[player] <=
          betting.total_committed[player]);
    CHECK(betting.stack[player] + betting.total_committed[player] ==
          initial_chips[player]);
  }
  CHECK(Pot(betting) ==
        betting.total_committed[0] + betting.total_committed[1]);

  CardMask seen = 0;
  for (Card card : state.board.cards()) {
    CHECK((seen & CardBit(card)) == 0);
    seen |= CardBit(card);
  }
  CHECK(seen == state.board.mask());
  CHECK(std::popcount(state.board.mask()) == state.board.count());
  CHECK(B(state).street == state.board.street());

  if (!std::holds_alternative<DecisionState>(state.betting) &&
      !std::holds_alternative<FoldTerminalState>(state.betting)) {
    CHECK(betting.street_committed[0] == betting.street_committed[1]);
  }
}

void CheckGeneratedRollout(uint32_t seed) {
  CAPTURE(seed);
  ExactPublicState state =
      MakeInitialState(kRules, {20, 20}, {1, 2});
  const std::array<Chips, kPlayerCount> initial_chips = {20, 20};
  const std::array<double, 3> sizes = {0.25, 0.5, 1.0};
  std::array<Card, kDeckCardCount> deck = {};
  std::copy(kDeck.begin(), kDeck.end(), deck.begin());
  std::mt19937 rng(seed);

  CheckGeneralInvariants(state, initial_chips);
  for (int step = 0; step < 64 && !IsTerminal(state); ++step) {
    if (std::holds_alternative<DecisionState>(state.betting)) {
      const std::vector<GameAction> menu = ActionsFor(state.betting, sizes);
      REQUIRE(menu.size() > 0);
      for (size_t i = 0; i < menu.size(); ++i) {
        ExactPublicState child = state;
        child.betting = Apply(state.betting, menu[i]);
        CheckGeneralInvariants(child, initial_chips);
        CHECK(child.board == state.board);
      }
      const size_t index = rng() % menu.size();
      state.betting = Apply(state.betting, menu[index]);
      CheckGeneralInvariants(state, initial_chips);
      continue;
    }

    std::shuffle(deck.begin(), deck.end(), rng);
    std::vector<Card> cards;
    for (Card card : deck) {
      if (!state.board.contains(card)) {
        cards.push_back(card);
      }
      if (cards.size() ==
          static_cast<size_t>(CardsForNextStreet(B(state).street))) {
        break;
      }
    }
    const BettingState before = state.betting;
    const CardMask board_before = state.board.mask();
    state = DealChance(state, cards, kRules);
    CheckGeneralInvariants(state, initial_chips);
    CHECK(B(state).stack == B(before).stack);
    CHECK(B(state).total_committed == B(before).total_committed);
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
  BettingData data;
  data.stack = stack;
  data.total_committed = total;
  data.street_committed = street;
  data.last_full_raise = last_full_raise;
  return DecisionState{data, static_cast<Player>(player)};
}

bool HasAction(const std::vector<GameAction>& menu, GameAction expected) {
  for (const GameAction& action : menu) {
    if (action == expected) {
      return true;
    }
  }
  return false;
}

void CheckMenu(const BettingState& state,
               absl::Span<const double> sizes) {
  const std::vector<GameAction> menu = ActionsFor(state, sizes);
  for (const GameAction& action : menu) {
    CHECK_NOTHROW(Apply(state, action));
  }
}

TEST_CASE("pot fractions use the pot after calling") {
  const std::array<double, 2> half_and_pot = {0.5, 1.0};
  const BettingState unopened =
      State({200, 200}, {50, 50}, {0, 0}, 0, 2);
  const std::vector<GameAction> bets = ActionsFor(unopened, half_and_pot);
  CHECK(HasAction(bets, {ActionKind::Bet, 50}));
  CHECK(HasAction(bets, {ActionKind::Bet, 100}));

  const BettingState fractional_pot =
      State({200, 200}, {5, 5}, {0, 0}, 0, 2);
  const std::array<double, 1> fraction = {0.21};
  const std::vector<GameAction> rounded_up =
      ActionsFor(fractional_pot, fraction);
  CHECK(HasAction(rounded_up, {ActionKind::Bet, 3}));

  const std::array<double, 1> pot = {1.0};
  const BettingState facing =
      State({200, 200}, {37, 63}, {0, 25}, 0, 25);
  const std::vector<GameAction> raises = ActionsFor(facing, pot);
  CHECK(HasAction(raises, {ActionKind::Raise, 150}));

  const BettingState big_blind =
      State({98, 98}, {2, 2}, {2, 2}, 1, 2);
  const std::vector<GameAction> options = ActionsFor(big_blind, pot);
  CHECK(HasAction(options, {ActionKind::Raise, 6}));
  CHECK_FALSE(HasAction(options, {ActionKind::Bet, 6}));

  const BettingState capped =
      State({50, 50}, {50, 50}, {0, 0}, 0, 2);
  const std::vector<GameAction> capped_actions =
      ActionsFor(capped, half_and_pot);
  CHECK_FALSE(HasAction(capped_actions, {ActionKind::Bet, 50}));
  CHECK(HasAction(capped_actions, {ActionKind::AllIn, 50}));
}

TEST_CASE("check-check completes a betting round") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 2, 2);
  B(state).street = StreetKind::Flop;
  state.betting = DecisionState{B(state), Player::B};
  B(state).street_committed = {0, 0};
  AddFlop(state.board, Flop());

  state.betting = Apply(state.betting, {ActionKind::Check});
  CHECK(std::holds_alternative<DecisionState>(state.betting));
  state.betting = Apply(state.betting, {ActionKind::Check});

  CHECK(std::holds_alternative<ChanceState>(state.betting));
  CHECK_FALSE(IsTerminal(state));
}

TEST_CASE("commitments update and reset across streets") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  const std::array<Chips, kPlayerCount> chips = {
      B(state).stack[0] + B(state).total_committed[0],
      B(state).stack[1] + B(state).total_committed[1],
  };

  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  CHECK(B(state).stack[0] == 18);
  CHECK(B(state).total_committed[0] == 2);
  CHECK(B(state).street_committed[0] == 2);
  CHECK(Pot(B(state)) == 4);
  state.betting = Apply(state.betting, {ActionKind::Check});
  state = DealChance(state, Flop(), kRules);

  CHECK(B(state).total_committed ==
        std::array<Chips, kPlayerCount>{2, 2});
  CHECK(B(state).street_committed ==
        std::array<Chips, kPlayerCount>{0, 0});
  CHECK(B(state).last_full_raise == kRules.minimum_bet);

  state.betting = Apply(state.betting, {ActionKind::Bet, 4});
  CHECK(B(state).stack[1] == 14);
  CHECK(B(state).total_committed[1] == 6);
  CHECK(B(state).street_committed[1] == 4);
  CHECK(B(state).last_full_raise == 4);
  CHECK(Pot(B(state)) == 8);
  for (size_t player = 0; player < kPlayerCount; ++player) {
    CHECK(B(state).stack[player] +
              B(state).total_committed[player] ==
          chips[player]);
  }
}

TEST_CASE("chip actions use final street commitments") {
  SUBCASE("raise") {
    ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
    state.betting = Apply(state.betting, {ActionKind::Raise, 5});

    CHECK(B(state).stack ==
          std::array<Chips, kPlayerCount>{15, 18});
    CHECK(B(state).total_committed ==
          std::array<Chips, kPlayerCount>{5, 2});
    CHECK(B(state).street_committed ==
          std::array<Chips, kPlayerCount>{5, 2});
    CHECK(Pot(B(state)) == 7);
  }

  SUBCASE("all-in") {
    ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
    state.betting = Apply(state.betting, {ActionKind::AllIn, 20});

    CHECK(B(state).stack ==
          std::array<Chips, kPlayerCount>{0, 18});
    CHECK(B(state).total_committed ==
          std::array<Chips, kPlayerCount>{20, 2});
    CHECK(B(state).street_committed ==
          std::array<Chips, kPlayerCount>{20, 2});
    CHECK(Pot(B(state)) == 22);
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
          {ActionKind::Raise, 3},
          {ActionKind::Raise, 4},
      },
      {
          "postflop opening bet",
          State({90, 90}, {10, 10}, {0, 0}, 0, 2),
          {ActionKind::Bet, 1},
          {ActionKind::Bet, 2},
      },
      {
          "raise after full raise",
          State({18, 14}, {2, 6}, {2, 6}, 0, 4),
          {ActionKind::Raise, 9},
          {ActionKind::Raise, 10},
      },
  }};

  for (const Case& test : cases) {
    CAPTURE(test.name);
    CHECK_THROWS_AS(Apply(test.state, test.too_small),
                    std::invalid_argument);
    CHECK_NOTHROW(Apply(test.state, test.minimum));
  }

  const BettingState preflop = cases[0].state;
  const std::array<double, 1> subminimum_size = {0.5};
  const std::vector<GameAction> menu = ActionsFor(preflop, subminimum_size);
  CHECK_FALSE(HasAction(menu, {ActionKind::Raise, 3}));
  CheckMenu(preflop, subminimum_size);
}

TEST_CASE("big blind retains the raise option after a limp") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  const std::array<double, 1> sizes = {0.5};
  const std::vector<GameAction> menu = ActionsFor(state.betting, sizes);

  CHECK(HasAction(menu, {ActionKind::Check, 0}));
  CHECK(HasAction(menu, {ActionKind::Raise, 4}));
  CHECK_FALSE(HasAction(menu, {ActionKind::Bet, 4}));
  CheckMenu(state.betting, sizes);
}

TEST_CASE("effective stacks and short all-ins bound aggression") {
  SUBCASE("full all-in raise") {
    const BettingState state =
        State({7, 20}, {8, 10}, {8, 10}, 0, 4);
    const BettingState child =
        Apply(state, {ActionKind::AllIn, 15});
    CHECK(B(child).street_committed[0] == 15);
    CHECK(B(child).last_full_raise == 5);
  }

  SUBCASE("short all-in raise") {
    const BettingState state =
        State({3, 20}, {8, 10}, {8, 10}, 0, 5);
    const std::array<double, 1> sizes = {1.0};
    const std::vector<GameAction> menu = ActionsFor(state, sizes);
    CHECK(HasAction(menu, {ActionKind::Call, 10}));
    CHECK(HasAction(menu, {ActionKind::AllIn, 11}));
    CHECK_FALSE(HasAction(menu, {ActionKind::Raise, 11}));

    const BettingState child =
        Apply(state, {ActionKind::AllIn, 11});
    CHECK(B(child).last_full_raise == 5);
    CHECK(B(child).actions_remaining == 1);
    CheckMenu(state, sizes);
  }

  SUBCASE("deeper stack cannot exceed the effective stack") {
    const BettingState state =
        State({100, 20}, {10, 10}, {0, 0}, 0, 2);
    const std::array<double, 2> sizes = {0.5, 2.0};
    const std::vector<GameAction> menu = ActionsFor(state, sizes);
    CHECK(HasAction(menu, {ActionKind::AllIn, 20}));
    for (const GameAction& action : menu) {
      CHECK(action.target_street_commitment <= 20);
    }

    const BettingState child =
        Apply(state, {ActionKind::AllIn, 20});
    CHECK(B(child).stack[0] == 80);
    CheckMenu(state, sizes);
  }
}

TEST_CASE("an all-in opponent cannot face new aggression") {
  const BettingState settled =
      State({20, 0}, {10, 10}, {0, 0}, 0, 2);
  const std::array<double, 1> sizes = {1.0};
  const std::vector<GameAction> settled_menu = ActionsFor(settled, sizes);
  CHECK(settled_menu.size() == 1);
  CHECK(HasAction(settled_menu, {ActionKind::Check, 0}));

  const BettingState facing_bet =
      State({20, 0}, {0, 10}, {0, 10}, 0, 2);
  const std::vector<GameAction> call_menu = ActionsFor(facing_bet, sizes);
  CHECK(HasAction(call_menu, {ActionKind::Fold, 0}));
  CHECK(HasAction(call_menu, {ActionKind::Call, 10}));
  CHECK_FALSE(HasAction(call_menu, {ActionKind::AllIn, 10}));

  const BettingState called =
      Apply(facing_bet, {ActionKind::Call, 10});
  CHECK(std::holds_alternative<ChanceState>(called));
  CheckMenu(facing_bet, sizes);
}

TEST_CASE("preflop all-in runout skips later decisions") {
  ExactPublicState state = test::InitialHeadsUpState(4, 20, 1, 2);
  state.betting = Apply(state.betting, {ActionKind::AllIn, 4});
  state.betting = Apply(state.betting, {ActionKind::Call, 4});
  REQUIRE(std::holds_alternative<ChanceState>(state.betting));

  state = DealChance(state, Flop(), kRules);
  CHECK(std::holds_alternative<ChanceState>(state.betting));
  CHECK_FALSE(IsTerminal(state));

  const std::array<Card, 1> turn = {
      C(9, Suit::Spades),
  };
  state = DealChance(state, turn, kRules);
  CHECK(std::holds_alternative<ChanceState>(state.betting));
  CHECK_FALSE(IsTerminal(state));

  const std::array<Card, 1> river = {
      C(3, Suit::Hearts),
  };
  state = DealChance(state, river, kRules);
  CHECK(std::holds_alternative<ShowdownState>(state.betting));
  CHECK(IsTerminal(state));
}

TEST_CASE("short all-in calls refund unmatched chips") {
  ExactPublicState state = test::InitialHeadsUpState(4, 20, 1, 2);
  B(state).stack = {3, 12};
  B(state).total_committed = {1, 8};
  B(state).street_committed = {1, 8};

  state.betting = Apply(state.betting, {ActionKind::Call, 4});

  CHECK(B(state).total_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(B(state).street_committed ==
        std::array<Chips, kPlayerCount>{4, 4});
  CHECK(B(state).stack ==
        std::array<Chips, kPlayerCount>{0, 16});
  CHECK(IsValidBettingData(B(state)));
}

TEST_CASE("fold completes the hand") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  state.betting = Apply(state.betting, {ActionKind::Fold});

  CHECK(std::holds_alternative<FoldTerminalState>(state.betting));
  CHECK(IsTerminal(state));
  CHECK(B(state).total_committed[0] !=
        B(state).total_committed[1]);
  CHECK(std::get<FoldTerminalState>(state.betting).folded == Player::A);
}

TEST_CASE("folds are terminal on every street") {
  const std::array<StreetKind, 4> streets = {
      StreetKind::Preflop,
      StreetKind::Flop,
      StreetKind::Turn,
      StreetKind::River,
  };
  for (StreetKind street : streets) {
    ExactPublicState state = ClosedState(street);
    state.betting = FoldTerminalState{B(state), Player::A};
    CHECK(IsTerminal(state));
    CHECK(TerminalUtility(std::get<FoldTerminalState>(state.betting),
                          Player::A) ==
          doctest::Approx(-10.0));
  }
}

TEST_CASE("river terminal utility handles win, loss, and tie") {
  const ComboId player0 = H(14, S::Hearts, 13, S::Hearts);
  const ComboId player1 = H(9, S::Hearts, 8, S::Hearts);
  const ExactPublicState win = Showdown({
      C(10, S::Hearts),
      C(11, S::Hearts),
      C(12, S::Hearts),
      C(2, S::Clubs),
      C(3, S::Diamonds),
  });
  REQUIRE(B(win).total_committed[0] ==
          B(win).total_committed[1]);
  const ShowdownState& showdown =
      std::get<ShowdownState>(win.betting);
  const double win_utility = TerminalUtility(
      showdown, win.board, player0, player1);
  const double loss_utility = TerminalUtility(
      showdown, win.board, player1, player0);
  CHECK(win_utility == doctest::Approx(10.0));
  CHECK(loss_utility == doctest::Approx(-10.0));
  CHECK(win_utility + loss_utility == doctest::Approx(0.0));

  const ExactPublicState tie = Showdown({
      C(2, S::Hearts),
      C(3, S::Diamonds),
      C(4, S::Clubs),
      C(5, S::Spades),
      C(6, S::Hearts),
  });
  REQUIRE(B(tie).total_committed[0] ==
          B(tie).total_committed[1]);
  CHECK(TerminalUtility(
            std::get<ShowdownState>(tie.betting),
            tie.board,
            H(14, S::Clubs, 13, S::Diamonds),
            H(12, S::Clubs, 11, S::Diamonds)) ==
        doctest::Approx(0.0));
}

TEST_CASE("a complete normal hand preserves exact state") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  Board board;
  CheckExactState("blinds", state, {19, 18}, {1, 2}, {1, 2}, 2,
                  StreetKind::Preflop, 0, 2, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  CheckExactState("small blind calls", state, {18, 18}, {2, 2},
                  {2, 2}, 2, StreetKind::Preflop, 1, 1,
                  board, StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Check});
  CheckExactState("big blind checks", state, {18, 18}, {2, 2},
                  {2, 2}, 2, StreetKind::Preflop, -1, 0, board,
                  StatePhase::Chance);

  state = DealChance(state, Flop(), kRules);
  AddFlop(board, Flop());
  CheckExactState("flop", state, {18, 18}, {2, 2}, {0, 0}, 2,
                  StreetKind::Flop, 1, 2, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Check});
  CheckExactState("flop check", state, {18, 18}, {2, 2}, {0, 0}, 2,
                  StreetKind::Flop, 0, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Bet, 2});
  CheckExactState("flop bet", state, {16, 18}, {4, 2}, {2, 0}, 2,
                  StreetKind::Flop, 1, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  CheckExactState("flop call", state, {16, 16}, {4, 4}, {2, 2}, 2,
                  StreetKind::Flop, -1, 0, board,
                  StatePhase::Chance);

  const std::array<Card, 1> turn = {C(9, S::Spades)};
  state = DealChance(state, turn, kRules);
  AddTurn(board, turn[0]);
  CheckExactState("turn", state, {16, 16}, {4, 4}, {0, 0}, 2,
                  StreetKind::Turn, 1, 2, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Check});
  CheckExactState("turn first check", state, {16, 16}, {4, 4},
                  {0, 0}, 2, StreetKind::Turn, 0, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Check});
  CheckExactState("turn second check", state, {16, 16}, {4, 4},
                  {0, 0}, 2, StreetKind::Turn, -1, 0, board,
                  StatePhase::Chance);

  const std::array<Card, 1> river = {C(3, S::Hearts)};
  state = DealChance(state, river, kRules);
  AddRiver(board, river[0]);
  CheckExactState("river", state, {16, 16}, {4, 4}, {0, 0}, 2,
                  StreetKind::River, 1, 2, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Bet, 2});
  CheckExactState("river bet", state, {16, 14}, {4, 6}, {0, 2}, 2,
                  StreetKind::River, 0, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  CheckExactState("river call", state, {14, 14}, {6, 6}, {2, 2}, 2,
                  StreetKind::River, -1, 0, board,
                  StatePhase::Terminal);

  const ComboId aces = H(14, S::Clubs, 14, S::Diamonds);
  const ComboId kings = H(13, S::Clubs, 13, S::Diamonds);
  CHECK(TerminalUtility(
            std::get<ShowdownState>(state.betting),
            state.board, aces, kings) ==
        doctest::Approx(6.0));
}

TEST_CASE("full raises update the minimum re-raise increment") {
  ExactPublicState state = test::InitialHeadsUpState(20, 20, 1, 2);
  const Board board;

  state.betting = Apply(state.betting, {ActionKind::Raise, 4});
  CheckExactState("minimum raise", state, {16, 18}, {4, 2}, {4, 2}, 2,
                  StreetKind::Preflop, 1, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Raise, 8});
  CheckExactState("full re-raise", state, {16, 12}, {4, 8}, {4, 8}, 4,
                  StreetKind::Preflop, 0, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Call, 8});
  CheckExactState("call re-raise", state, {12, 12}, {8, 8}, {8, 8}, 4,
                  StreetKind::Preflop, -1, 0, board,
                  StatePhase::Chance);
}

TEST_CASE("a short all-in raise preserves the full-raise increment") {
  ExactPublicState state = test::InitialHeadsUpState(20, 5, 1, 2);
  Board board;

  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  state.betting = Apply(state.betting, {ActionKind::Check});
  state = DealChance(state, Flop(), kRules);
  AddFlop(board, Flop());
  CheckExactState("short stack reaches flop", state, {18, 3}, {2, 2},
                  {0, 0}, 2, StreetKind::Flop, 1, 2,
                  board, StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Check});
  state.betting = Apply(state.betting, {ActionKind::Bet, 2});
  CheckExactState("player zero bets", state, {16, 3}, {4, 2}, {2, 0}, 2,
                  StreetKind::Flop, 1, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::AllIn, 3});
  CheckExactState("subminimum all-in raise", state, {16, 0}, {4, 5},
                  {2, 3}, 2, StreetKind::Flop, 0, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Call, 3});
  CheckExactState("short raise called", state, {15, 0}, {5, 5}, {3, 3},
                  2, StreetKind::Flop, -1, 0, board,
                  StatePhase::Chance);

  const std::array<Card, 1> turn = {C(9, S::Spades)};
  state = DealChance(state, turn, kRules);
  AddTurn(board, turn[0]);
  CheckExactState("automatic turn", state, {15, 0}, {5, 5}, {0, 0}, 2,
                  StreetKind::Turn, -1, 2, board,
                  StatePhase::Chance);

  const std::array<Card, 1> river = {C(3, S::Hearts)};
  state = DealChance(state, river, kRules);
  AddRiver(board, river[0]);
  CheckExactState("automatic river", state, {15, 0}, {5, 5}, {0, 0}, 2,
                  StreetKind::River, -1, 2, board,
                  StatePhase::Terminal);
}

TEST_CASE("effective stacks leave unmatched chips uncommitted") {
  ExactPublicState state = test::InitialHeadsUpState(100, 20, 1, 2);
  Board board;
  state.betting = Apply(state.betting, {ActionKind::Call, 2});
  state.betting = Apply(state.betting, {ActionKind::Check});
  state = DealChance(state, Flop(), kRules);
  AddFlop(board, Flop());
  state.betting = Apply(state.betting, {ActionKind::Check});

  const std::array<double, 2> sizes = {1.0, 10.0};
  const std::vector<GameAction> menu = ActionsFor(state.betting, sizes);
  for (const GameAction& action : menu) {
    CHECK(action.target_street_commitment <= 18);
  }

  state.betting = Apply(state.betting, {ActionKind::AllIn, 18});
  CheckExactState("effective all-in", state, {80, 18}, {20, 2}, {18, 0},
                  18, StreetKind::Flop, 1, 1, board,
                  StatePhase::Decision);

  state.betting = Apply(state.betting, {ActionKind::Call, 18});
  CheckExactState("effective all-in called", state, {80, 0}, {20, 20},
                  {18, 18}, 18, StreetKind::Flop, -1, 0, board,
                  StatePhase::Chance);
}

TEST_CASE("solver action count follows configured pot fractions") {
  const BettingState state =
      State({1000, 1000}, {100, 100}, {0, 0}, 0, 2);
  const std::array<double, 9> sizes = {
      0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18};

  const std::vector<GameAction> actions = ActionsFor(state, sizes);

  CHECK(actions.size() == 11);
  for (const GameAction& action : actions) {
    CHECK_NOTHROW(Apply(state, action));
  }
}

}  // namespace
}  // namespace poker
