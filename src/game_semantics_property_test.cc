#include "src/betting_abstraction.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"
#include "src/card_utils.h"
#include "src/combo.h"
#include "src/game_tree.h"
#include "src/hand_evaluator.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace poker {
namespace {

SolverConfig TestConfig() {
  SolverConfig config;
  config.starting_stack_size = 20;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {0.5, 1.0};
  config.chance_samples = 1;
  return config;
}

CompactPublicState InitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;

  CompactPublicState state;
  state.stack[0] = std::max(0, config.starting_stack_size - small_blind);
  state.stack[1] = std::max(0, config.starting_stack_size - big_blind);
  state.pot = small_blind + big_blind;
  state.street = StreetKind::kPreflop;
  state.folded_player = -1;
  state.player_to_act = 0;
  state.player_contribution = {small_blind, big_blind};
  return state;
}

bool CompactTerminal(const CompactPublicState& state) {
  const ExactGameState exact = ExactGameStateFromCompact(state);
  return IsTerminal(exact.betting, exact.board);
}

int CompactPlayerToAct(const CompactPublicState& state) {
  const ExactGameState exact = ExactGameStateFromCompact(state);
  return GetPlayerToAct(exact.betting, exact.board);
}

bool CompactBettingRoundOver(const CompactPublicState& state) {
  return IsBettingRoundOver(BettingStateFromCompact(state));
}

CompactPublicState CompactApplyChance(const CompactPublicState& state,
                                      absl::Span<const CardId> cards) {
  return ToCompact(ApplyChance(ExactGameStateFromCompact(state), cards));
}

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

std::string CardString(CardId card) {
  static constexpr std::array<char, 13> kRanks = {
      '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'};
  static constexpr std::array<char, 4> kSuits = {'h', 'd', 'c', 's'};
  std::string out;
  out.push_back(kRanks[static_cast<size_t>(RankFromCardId(card) - 2)]);
  out.push_back(kSuits[static_cast<size_t>(SuitIndex(SuitFromCardId(card)))]);
  return out;
}

std::string ActionString(const GameAction& action) {
  std::ostringstream out;
  out << "player=" << action.player << " kind="
      << static_cast<int>(action.kind) << " amount=" << action.amount;
  return out.str();
}

std::string StateString(const CompactPublicState& state) {
  std::ostringstream out;
  out << "street=" << static_cast<int>(state.street)
      << " pot=" << state.pot
      << " stack={" << state.stack[0] << "," << state.stack[1] << "}"
      << " contrib={" << state.player_contribution[0] << ","
      << state.player_contribution[1] << "}"
      << " player_to_act=" << state.player_to_act
      << " folded=" << state.folded_player
      << " all_in=" << state.all_in
      << " board=";
  for (uint8_t i = 0; i < state.board_count; ++i) {
    out << CardString(state.board_cards[static_cast<size_t>(i)]);
  }
  return out.str();
}

struct ReachableTrace {
  uint32_t seed = 0;
  int max_steps = 0;
  SolverConfig config;
  std::vector<std::string> steps;

  std::string dump() const {
    std::ostringstream out;
    out << "seed=" << seed << " max_steps=" << max_steps
        << " stack=" << config.starting_stack_size
        << " blinds=" << config.small_blind << "/" << config.big_blind
        << "\n";
    for (const std::string& step : steps) {
      out << step << "\n";
    }
    return out.str();
  }
};

void Require(bool condition,
             const ReachableTrace& trace,
             const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message + "\n" + trace.dump());
  }
}

int TotalChips(const CompactPublicState& state) {
  return state.pot + state.stack[0] + state.stack[1];
}

void ValidateBoard(const CompactPublicState& state,
                   const ReachableTrace& trace) {
  Require(std::popcount(state.board_mask) == state.board_count, trace,
          "board mask popcount must equal board count");

  CardMask seen = 0;
  for (uint8_t i = 0; i < state.board_count; ++i) {
    const CardId card = state.board_cards[static_cast<size_t>(i)];
    const CardMask bit = CardBit(card);
    Require((seen & bit) == 0, trace, "board cards must be unique");
    Require((state.board_mask & bit) != 0, trace,
            "board card must be present in mask");
    seen |= bit;
  }

  Require(BoardCardsForStreet(state.street) == state.board_count, trace,
          "street and board count must agree");
}

void ValidateState(const CompactPublicState& state,
                   int total_chips,
                   const ReachableTrace& trace) {
  Require(state.pot >= 0, trace, "pot must be non-negative");
  Require(state.stack[0] >= 0, trace, "player A stack must be non-negative");
  Require(state.stack[1] >= 0, trace, "player B stack must be non-negative");
  Require(state.player_contribution[0] >= 0, trace,
          "player A contribution must be non-negative");
  Require(state.player_contribution[1] >= 0, trace,
          "player B contribution must be non-negative");
  Require(TotalChips(state) == total_chips, trace,
          "pot plus stacks must conserve chips");
  Require(state.pot == state.player_contribution[0] +
                           state.player_contribution[1],
          trace, "pot must equal total contributions");
  ValidateBoard(state, trace);
}

void ValidateActionTransition(const CompactPublicState& parent,
                              const CompactPublicState& child,
                              int player,
                              int total_chips,
                              const ReachableTrace& trace) {
  ValidateState(child, total_chips, trace);
  const int committed = parent.stack[player] - child.stack[player];
  Require(committed >= 0, trace, "action cannot add chips to stack");
  Require(child.pot == parent.pot + committed, trace,
          "pot must increase by committed chips");
  Require(child.player_contribution[player] ==
              parent.player_contribution[player] + committed,
          trace, "acting player contribution must increase by committed chips");
  Require(child.player_contribution[Opponent(player)] ==
              parent.player_contribution[Opponent(player)],
          trace, "opponent contribution must not change");
}

CompactPublicState ApplyChecked(const CompactPublicState& parent,
                                const GameAction& action,
                                int total_chips,
                                const ReachableTrace& trace) {
  const int player = CompactPlayerToAct(parent);
  Require(IsPlayer(player), trace, "action transition needs an acting player");
  const CompactPublicState child = ApplyAction(parent, action);
  ValidateActionTransition(parent, child, player, total_chips, trace);
  return child;
}

void ValidateChanceTransition(const CompactPublicState& parent,
                              const CompactPublicState& child,
                              absl::Span<const CardId> cards,
                              int total_chips,
                              const ReachableTrace& trace) {
  ValidateState(child, total_chips, trace);
  Require(TotalChips(child) == TotalChips(parent), trace,
          "chance cannot change chips");
  Require(child.street == StreetAfterChance(parent.street), trace,
          "chance must advance street");
  Require(child.board_count == BoardCardsForStreet(child.street), trace,
          "chance must produce exact board count for street");
  Require(child.actions_this_street == 0, trace,
          "chance must reset street action count");

  CardMask dealt = 0;
  for (CardId card : cards) {
    const CardMask bit = CardBit(card);
    Require((parent.board_mask & bit) == 0, trace,
            "chance card cannot overlap old board");
    Require((dealt & bit) == 0, trace, "chance cards must be unique");
    dealt |= bit;
  }
}

std::vector<GameAction> LegalActions(const BettingAbstraction& betting,
                                     const CompactPublicState& state) {
  const auto menu = betting.actions_for_betting_node(
      BettingStateFromCompact(state), state.player_to_act);
  return std::vector<GameAction>(menu.actions.begin(),
                                 menu.actions.begin() + menu.count);
}

void TraverseBettingTree(const BettingAbstraction& betting,
                         const CompactPublicState& state,
                         int total_chips,
                         int depth,
                         int max_depth,
                         ReachableTrace trace) {
  ValidateState(state, total_chips, trace);
  if (depth >= max_depth || CompactTerminal(state) ||
      CompactBettingRoundOver(state)) {
    return;
  }

  const int player = CompactPlayerToAct(state);
  Require(IsPlayer(player), trace, "betting tree node must have a player");
  const std::vector<GameAction> actions = LegalActions(betting, state);
  Require(!actions.empty(), trace, "betting tree node must have actions");

  for (const GameAction& action : actions) {
    CompactPublicState child =
        ApplyChecked(state, action, total_chips, trace);
    trace.steps.push_back("depth " + std::to_string(depth) + " " +
                          ActionString(action) + "\n  before " +
                          StateString(state) + "\n  after  " +
                          StateString(child));
    TraverseBettingTree(betting, child, total_chips, depth + 1, max_depth,
                        trace);
  }
}

void CheckReachableCase(uint32_t seed, int max_steps) {
  SolverConfig config = TestConfig();
  config.starting_stack_size = 8 + static_cast<int>(seed % 24);
  config.big_blind = std::min(2, config.starting_stack_size / 2);
  config.small_blind = std::max(1, config.big_blind / 2);

  ReachableTrace trace{seed, max_steps, config, {}};
  std::mt19937 rng(seed);
  BettingAbstraction betting(config);
  CompactPublicState state = InitialState(config);
  const int total_chips = TotalChips(state);
  ValidateState(state, total_chips, trace);

  for (int step = 0; step < max_steps && !CompactTerminal(state); ++step) {
    const int player = CompactPlayerToAct(state);
    if (IsPlayer(player)) {
      const std::vector<GameAction> actions = LegalActions(betting, state);
      Require(!actions.empty(), trace, "player node must have legal actions");
      for (const GameAction& action : actions) {
        const CompactPublicState child = ApplyAction(state, action);
        ValidateActionTransition(state, child, player, total_chips, trace);
      }

      std::uniform_int_distribution<size_t> choose(0, actions.size() - 1);
      const GameAction action = actions[choose(rng)];
      const CompactPublicState parent = state;
      state = ApplyAction(state, action);
      trace.steps.push_back("action " + ActionString(action) + "\n  before " +
                            StateString(parent) + "\n  after  " +
                            StateString(state));
      ValidateActionTransition(parent, state, player, total_chips, trace);
    } else {
      const CompactPublicState parent = state;
      const auto cards = SampleStreetCards(state, 0, rng);
      state = CompactApplyChance(state, cards);
      std::ostringstream step_text;
      step_text << "chance";
      for (CardId card : cards) {
        step_text << " " << CardString(card);
      }
      step_text << "\n  before " << StateString(parent)
                << "\n  after  " << StateString(state);
      trace.steps.push_back(step_text.str());
      ValidateChanceTransition(parent, state, cards, total_chips, trace);
    }
  }
}

CompactPublicState FlopState() {
  CompactPublicState state;
  state.stack = {18, 18};
  state.pot = 4;
  state.street = StreetKind::kFlop;
  state.player_to_act = 1;
  state.folded_player = -1;
  state.player_contribution = {2, 2};
  AddBoardCard(state, MakeCardId(2, SuitKind::kHearts));
  AddBoardCard(state, MakeCardId(7, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(12, SuitKind::kClubs));
  return state;
}

CompactPublicState RiverState() {
  CompactPublicState state;
  state.stack = {20, 20};
  state.pot = 20;
  state.street = StreetKind::kRiver;
  state.player_to_act = 1;
  state.folded_player = -1;
  state.player_contribution = {10, 10};
  AddBoardCard(state, MakeCardId(2, SuitKind::kHearts));
  AddBoardCard(state, MakeCardId(7, SuitKind::kDiamonds));
  AddBoardCard(state, MakeCardId(12, SuitKind::kClubs));
  AddBoardCard(state, MakeCardId(9, SuitKind::kSpades));
  AddBoardCard(state, MakeCardId(3, SuitKind::kHearts));
  return state;
}

void CheckCompletedRound(const CompactPublicState& state,
                         bool terminal) {
  CAPTURE(StateString(state));
  CHECK(CompactBettingRoundOver(state));
  CHECK(CompactPlayerToAct(state) == -1);
  CHECK(CompactTerminal(state) == terminal);
}

CompactPublicState RiverShowdown(std::initializer_list<CardId> board) {
  CompactPublicState state;
  state.stack = {0, 0};
  state.pot = 20;
  state.street = StreetKind::kRiver;
  state.player_to_act = -1;
  state.folded_player = -1;
  state.player_contribution = {10, 10};
  for (CardId card : board) {
    AddBoardCard(state, card);
  }
  return state;
}

double OracleUtility(const CompactPublicState& state,
                     ComboId a_hand,
                     ComboId b_hand,
                     int player) {
  const double contribution = state.player_contribution[player];
  if (state.folded_player == player) {
    return -contribution;
  }
  if (state.folded_player == Opponent(player)) {
    return state.pot - contribution;
  }

  HandEvaluator evaluator;
  const int comparison = evaluator.compare_hands(a_hand, b_hand, state);
  const int player_comparison = player == 0 ? comparison : -comparison;
  if (player_comparison > 0) {
    return state.pot - contribution;
  }
  if (player_comparison < 0) {
    return -contribution;
  }
  return state.pot / 2.0 - contribution;
}

void CheckUtility(const CompactPublicState& state,
                  ComboId a_hand,
                  ComboId b_hand,
                  double expected_a) {
  CAPTURE(expected_a);
  CAPTURE(StateString(state));
  const double actual_a =
      GetUtility(ExactGameStateFromCompact(state), a_hand, b_hand);
  const double oracle_a = OracleUtility(state, a_hand, b_hand, 0);
  const double oracle_b = OracleUtility(state, a_hand, b_hand, 1);
  CHECK(actual_a == doctest::Approx(expected_a));
  CHECK(actual_a == doctest::Approx(oracle_a));
  CHECK(oracle_a + oracle_b == doctest::Approx(0.0));
}

TEST_CASE("reachable random states preserve poker invariants") {
  const bool rapidcheck_ok = rc::check("reachable state invariants", [] {
    const uint32_t seed = *rc::gen::arbitrary<uint32_t>();
    const int max_steps = *rc::gen::inRange(1, 64);
    try {
      CheckReachableCase(seed, max_steps);
    } catch (const std::exception& error) {
      RC_LOG() << error.what();
      RC_ASSERT(false);
    }
  });
  CHECK(rapidcheck_ok);

  for (uint32_t seed = 0; seed < 2000; ++seed) {
    CAPTURE(seed);
    CHECK_NOTHROW(CheckReachableCase(seed, 64));
  }
}

TEST_CASE("representative invalid actions are rejected") {
  const SolverConfig config = TestConfig();
  const CompactPublicState root = InitialState(config);
  CHECK_THROWS(ApplyAction(root, {ActionKind::kCheck, 0, -1}));
  CHECK_THROWS(ApplyAction(root, {ActionKind::kRaise, 1, -1}));
  CHECK_THROWS(ApplyAction(root, {ActionKind::kRaise, root.stack[0], -1}));

  const CompactPublicState flop = FlopState();
  CHECK_THROWS(ApplyAction(flop, {ActionKind::kCall, 0, -1}));
  CHECK_THROWS(ApplyAction(flop, {ActionKind::kBet, flop.stack[1], -1}));

  const CompactPublicState folded =
      ApplyAction(root, {ActionKind::kFold, 0, -1});
  CHECK_THROWS(ApplyAction(folded, {ActionKind::kCall, 1, -1}));

  CompactPublicState broke = flop;
  broke.stack[1] = 0;
  broke.player_to_act = 1;
  CHECK_THROWS(ApplyAction(broke, {ActionKind::kCheck, 0, -1}));
}

TEST_CASE("deterministic action transitions preserve chip accounting") {
  const SolverConfig config = TestConfig();
  ReachableTrace trace{0, 0, config, {}};
  const CompactPublicState root = InitialState(config);
  const int total_chips = TotalChips(root);
  ValidateState(root, total_chips, trace);
  CHECK(root.street == StreetKind::kPreflop);
  CHECK(root.player_to_act == 0);
  CHECK(root.player_contribution[0] == config.small_blind);
  CHECK(root.player_contribution[1] == config.big_blind);

  CompactPublicState call_check =
      ApplyChecked(root, {ActionKind::kCall, 1}, total_chips, trace);
  call_check = ApplyChecked(call_check, {ActionKind::kCheck}, total_chips,
                            trace);
  CheckCompletedRound(call_check, false);

  CompactPublicState fold =
      ApplyChecked(root, {ActionKind::kFold}, total_chips, trace);
  CheckCompletedRound(fold, true);

  CompactPublicState raise_call =
      ApplyChecked(root, {ActionKind::kRaise, 4}, total_chips, trace);
  raise_call = ApplyChecked(raise_call, {ActionKind::kCall}, total_chips,
                            trace);
  CheckCompletedRound(raise_call, false);

  CompactPublicState check_bet_call =
      ApplyChecked(FlopState(), {ActionKind::kCheck}, 40, trace);
  check_bet_call =
      ApplyChecked(check_bet_call, {ActionKind::kBet, 2}, 40, trace);
  check_bet_call =
      ApplyChecked(check_bet_call, {ActionKind::kCall}, 40, trace);
  CheckCompletedRound(check_bet_call, false);

  CompactPublicState bet_raise_call =
      ApplyChecked(FlopState(), {ActionKind::kBet, 4}, 40, trace);
  bet_raise_call =
      ApplyChecked(bet_raise_call, {ActionKind::kRaise, 8}, 40, trace);
  bet_raise_call =
      ApplyChecked(bet_raise_call, {ActionKind::kCall}, 40, trace);
  CheckCompletedRound(bet_raise_call, false);

  CompactPublicState all_in =
      ApplyChecked(FlopState(), {ActionKind::kAllIn}, 40, trace);
  all_in = ApplyChecked(all_in, {ActionKind::kCall}, 40, trace);
  CheckCompletedRound(all_in, false);

  CompactPublicState short_call = root;
  short_call.stack[0] = 3;
  short_call.stack[1] = 12;
  short_call.pot = 9;
  short_call.player_contribution = {1, 8};
  short_call = ApplyChecked(short_call, {ActionKind::kCall}, 24, trace);
  CheckCompletedRound(short_call, false);
  CHECK(short_call.stack[0] == 0);
  CHECK(short_call.player_contribution[0] < short_call.player_contribution[1]);

  CompactPublicState river_check =
      ApplyChecked(RiverState(), {ActionKind::kCheck}, 60, trace);
  river_check = ApplyChecked(river_check, {ActionKind::kCheck}, 60, trace);
  CheckCompletedRound(river_check, true);

  CompactPublicState river_call =
      ApplyChecked(RiverState(), {ActionKind::kBet, 10}, 60, trace);
  river_call = ApplyChecked(river_call, {ActionKind::kCall}, 60, trace);
  CheckCompletedRound(river_call, true);

  CompactPublicState river_fold =
      ApplyChecked(RiverState(), {ActionKind::kBet, 10}, 60, trace);
  river_fold = ApplyChecked(river_fold, {ActionKind::kFold}, 60, trace);
  CheckCompletedRound(river_fold, true);
}

TEST_CASE("tiny betting tree is exhaustively valid to bounded depth") {
  SolverConfig config = TestConfig();
  config.starting_stack_size = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  config.bet_sizes = {0.5, 1.0};

  const CompactPublicState root = InitialState(config);
  const int total_chips = TotalChips(root);
  ReachableTrace trace{0, 8, config, {}};
  TraverseBettingTree(BettingAbstraction(config), root, total_chips, 0, 8,
                      trace);
}

TEST_CASE("betting-round completion cases agree") {
  const SolverConfig config = TestConfig();
  const CompactPublicState root = InitialState(config);
  CompactPublicState preflop_complete =
      ApplyAction(root, {ActionKind::kCall, 1, -1});
  CHECK(!CompactBettingRoundOver(preflop_complete));
  preflop_complete = ApplyAction(preflop_complete, {ActionKind::kCheck});
  CheckCompletedRound(preflop_complete, false);

  CompactPublicState checked = ApplyAction(FlopState(), {ActionKind::kCheck});
  checked = ApplyAction(checked, {ActionKind::kCheck});
  CheckCompletedRound(checked, false);

  CompactPublicState bet_call = ApplyAction(FlopState(), {ActionKind::kBet, 4});
  bet_call = ApplyAction(bet_call, {ActionKind::kCall, 4});
  CheckCompletedRound(bet_call, false);

  CompactPublicState bet_fold = ApplyAction(FlopState(), {ActionKind::kBet, 4});
  bet_fold = ApplyAction(bet_fold, {ActionKind::kFold});
  CheckCompletedRound(bet_fold, true);

  CompactPublicState raise_call =
      ApplyAction(root, {ActionKind::kRaise, 4, -1});
  raise_call = ApplyAction(raise_call, {ActionKind::kCall, 3, -1});
  CheckCompletedRound(raise_call, false);

  CompactPublicState short_call = root;
  short_call.stack[0] = 3;
  short_call.stack[1] = 12;
  short_call.pot = 9;
  short_call.player_contribution = {1, 8};
  short_call = ApplyAction(short_call, {ActionKind::kCall, 7, -1});
  CheckCompletedRound(short_call, false);
  CHECK(short_call.stack[0] == 0);
  CHECK(short_call.player_contribution[0] < short_call.player_contribution[1]);

  CompactPublicState all_in = ApplyAction(FlopState(), {ActionKind::kAllIn});
  all_in = ApplyAction(all_in, {ActionKind::kCall});
  CheckCompletedRound(all_in, false);

  CompactPublicState runout = ApplyAction(root, {ActionKind::kCall, 1, -1});
  runout = ApplyAction(runout, {ActionKind::kCheck});
  runout = CompactApplyChance(runout, {MakeCardId(2, SuitKind::kHearts),
                                       MakeCardId(7, SuitKind::kDiamonds),
                                       MakeCardId(12, SuitKind::kClubs)});
  runout = ApplyAction(runout, {ActionKind::kAllIn});
  runout = ApplyAction(runout, {ActionKind::kCall});
  runout = CompactApplyChance(runout, {MakeCardId(9, SuitKind::kSpades)});
  runout = CompactApplyChance(runout, {MakeCardId(3, SuitKind::kHearts)});
  CHECK(CompactTerminal(runout));
  CHECK(CompactPlayerToAct(runout) == -1);
}

TEST_CASE("chance sampling excludes known cards") {
  CompactPublicState state = InitialState(TestConfig());
  state = ApplyAction(state, {ActionKind::kCall, 1, -1});

  const ComboId a_hand =
      Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const ComboId b_hand =
      Combo(12, SuitKind::kClubs, 11, SuitKind::kClubs);
  const CardMask blocked = ComboMask(a_hand) | ComboMask(b_hand);
  std::mt19937 rng(123);
  for (int i = 0; i < 500; ++i) {
    const auto cards = SampleStreetCards(state, blocked, rng);
    CHECK(cards.size() == 3);
    for (CardId card : cards) {
      CHECK((blocked & CardBit(card)) == 0);
    }
  }
}

TEST_CASE("terminal utility matches independent oracle") {
  CompactPublicState folded = InitialState(TestConfig());
  folded = ApplyAction(folded, {ActionKind::kFold});
  CheckUtility(folded,
               Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts),
               Combo(12, SuitKind::kClubs, 11, SuitKind::kClubs),
               -1.0);

  CheckUtility(RiverShowdown({MakeCardId(14, SuitKind::kHearts),
                              MakeCardId(13, SuitKind::kHearts),
                              MakeCardId(12, SuitKind::kHearts),
                              MakeCardId(11, SuitKind::kHearts),
                              MakeCardId(2, SuitKind::kClubs)}),
               Combo(10, SuitKind::kHearts, 4, SuitKind::kClubs),
               Combo(9, SuitKind::kHearts, 5, SuitKind::kClubs),
               10.0);

  CheckUtility(RiverShowdown({MakeCardId(14, SuitKind::kHearts),
                              MakeCardId(14, SuitKind::kDiamonds),
                              MakeCardId(13, SuitKind::kHearts),
                              MakeCardId(7, SuitKind::kHearts),
                              MakeCardId(2, SuitKind::kHearts)}),
               Combo(13, SuitKind::kClubs, 13, SuitKind::kSpades),
               Combo(12, SuitKind::kHearts, 11, SuitKind::kHearts),
               10.0);

  CheckUtility(RiverShowdown({MakeCardId(2, SuitKind::kHearts),
                              MakeCardId(3, SuitKind::kDiamonds),
                              MakeCardId(4, SuitKind::kClubs),
                              MakeCardId(9, SuitKind::kHearts),
                              MakeCardId(13, SuitKind::kSpades)}),
               Combo(14, SuitKind::kClubs, 5, SuitKind::kDiamonds),
               Combo(12, SuitKind::kClubs, 12, SuitKind::kDiamonds),
               10.0);

  CheckUtility(RiverShowdown({MakeCardId(2, SuitKind::kHearts),
                              MakeCardId(3, SuitKind::kDiamonds),
                              MakeCardId(4, SuitKind::kClubs),
                              MakeCardId(5, SuitKind::kSpades),
                              MakeCardId(6, SuitKind::kHearts)}),
               Combo(14, SuitKind::kClubs, 13, SuitKind::kDiamonds),
               Combo(12, SuitKind::kClubs, 11, SuitKind::kDiamonds),
               0.0);

  CheckUtility(RiverShowdown({MakeCardId(14, SuitKind::kHearts),
                              MakeCardId(14, SuitKind::kDiamonds),
                              MakeCardId(13, SuitKind::kClubs),
                              MakeCardId(13, SuitKind::kSpades),
                              MakeCardId(12, SuitKind::kHearts)}),
               Combo(2, SuitKind::kClubs, 2, SuitKind::kDiamonds),
               Combo(3, SuitKind::kClubs, 4, SuitKind::kDiamonds),
               0.0);

  CheckUtility(RiverShowdown({MakeCardId(2, SuitKind::kHearts),
                              MakeCardId(5, SuitKind::kHearts),
                              MakeCardId(9, SuitKind::kHearts),
                              MakeCardId(13, SuitKind::kHearts),
                              MakeCardId(12, SuitKind::kSpades)}),
               Combo(14, SuitKind::kHearts, 3, SuitKind::kClubs),
               Combo(13, SuitKind::kDiamonds, 12, SuitKind::kDiamonds),
               10.0);
}

}  // namespace
}  // namespace poker
