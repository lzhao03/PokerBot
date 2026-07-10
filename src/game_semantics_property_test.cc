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

ExactGameState InitialState(const SolverConfig& config) {
  const int small_blind = config.small_blind > 0 ? config.small_blind : 1;
  const int big_blind = config.big_blind > 0 ? config.big_blind : 2;

  ExactGameState state;
  state.betting.stack[0] =
      std::max(0, config.starting_stack_size - small_blind);
  state.betting.stack[1] =
      std::max(0, config.starting_stack_size - big_blind);
  state.betting.street = StreetKind::kPreflop;
  state.betting.folded_player = -1;
  state.betting.player_to_act = 0;
  state.betting.committed = {small_blind, big_blind};
  return state;
}

bool Terminal(const ExactGameState& state) {
  return IsTerminal(state.betting, state.board);
}

bool BettingRoundOver(const ExactGameState& state) {
  return IsBettingRoundOver(state.betting);
}

ExactGameState ApplyStateAction(ExactGameState state,
                                const GameAction& action) {
  state.betting = ApplyAction(state.betting, action);
  return state;
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
  out << "kind=" << static_cast<int>(action.kind)
      << " amount=" << action.amount;
  return out.str();
}

std::string StateString(const ExactGameState& state) {
  const BettingState& betting = state.betting;
  std::ostringstream out;
  out << "street=" << static_cast<int>(betting.street)
      << " pot=" << Pot(betting)
      << " stack={" << betting.stack[0] << "," << betting.stack[1] << "}"
      << " committed={" << betting.committed[0] << ","
      << betting.committed[1] << "}"
      << " player_to_act=" << static_cast<int>(betting.player_to_act)
      << " folded=" << static_cast<int>(betting.folded_player)
      << " all_in=" << AnyPlayerAllIn(betting)
      << " board=";
  for (CardId card : state.board.span()) {
    out << CardString(card);
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

int TotalChips(const ExactGameState& state) {
  return Pot(state.betting) + state.betting.stack[0] +
         state.betting.stack[1];
}

void ValidateBoard(const ExactGameState& state,
                   const ReachableTrace& trace) {
  Require(std::popcount(state.board.mask) == state.board.count, trace,
          "board mask popcount must equal board count");

  CardMask seen = 0;
  for (CardId card : state.board.span()) {
    const CardMask bit = CardBit(card);
    Require((seen & bit) == 0, trace, "board cards must be unique");
    Require((state.board.mask & bit) != 0, trace,
            "board card must be present in mask");
    seen |= bit;
  }

  Require(BoardCardsForStreet(state.betting.street) == state.board.count,
          trace,
          "street and board count must agree");
}

void ValidateState(const ExactGameState& state,
                   int total_chips,
                   const ReachableTrace& trace) {
  const BettingState& betting = state.betting;
  Require(Pot(betting) >= 0, trace, "pot must be non-negative");
  Require(betting.stack[0] >= 0, trace,
          "player A stack must be non-negative");
  Require(betting.stack[1] >= 0, trace,
          "player B stack must be non-negative");
  Require(betting.committed[0] >= 0, trace,
          "player A committed chips must be non-negative");
  Require(betting.committed[1] >= 0, trace,
          "player B committed chips must be non-negative");
  Require(TotalChips(state) == total_chips, trace,
          "pot plus stacks must conserve chips");
  Require(Pot(betting) == betting.committed[0] + betting.committed[1],
          trace, "pot must equal total committed chips");
  if (BettingRoundOver(state)) {
    Require(betting.player_to_act == -1, trace,
            "completed state must not have an acting player");
  } else {
    Require(IsPlayer(betting.player_to_act), trace,
            "decision state must have an acting player");
  }
  ValidateBoard(state, trace);
}

void ValidateActionTransition(const ExactGameState& parent,
                              const ExactGameState& child,
                              int player,
                              int total_chips,
                              const ReachableTrace& trace) {
  ValidateState(child, total_chips, trace);
  const int committed =
      parent.betting.stack[player] - child.betting.stack[player];
  Require(committed >= 0, trace, "action cannot add chips to stack");
  Require(Pot(child.betting) == Pot(parent.betting) + committed, trace,
          "pot must increase by committed chips");
  Require(child.betting.committed[player] ==
              parent.betting.committed[player] + committed,
          trace, "acting player's committed chips must increase");
  Require(child.betting.committed[Opponent(player)] ==
              parent.betting.committed[Opponent(player)],
          trace, "opponent's committed chips must not change");
  Require(child.board.mask == parent.board.mask, trace,
          "action cannot change the board");
}

ExactGameState ApplyChecked(const ExactGameState& parent,
                            const GameAction& action,
                            int total_chips,
                            const ReachableTrace& trace) {
  const int player = parent.betting.player_to_act;
  Require(IsPlayer(player), trace, "action transition needs an acting player");
  ExactGameState child = parent;
  child.betting = ApplyAction(parent.betting, action);
  ValidateActionTransition(parent, child, player, total_chips, trace);
  return child;
}

void ValidateChanceTransition(const ExactGameState& parent,
                              const ExactGameState& child,
                              absl::Span<const CardId> cards,
                              int total_chips,
                              const ReachableTrace& trace) {
  ValidateState(child, total_chips, trace);
  Require(TotalChips(child) == TotalChips(parent), trace,
          "chance cannot change chips");
  Require(child.betting.street == StreetAfterChance(parent.betting.street),
          trace,
          "chance must advance street");
  Require(child.board.count == BoardCardsForStreet(child.betting.street),
          trace,
          "chance must produce exact board count for street");
  Require(child.betting.pending_action_mask == kAllPlayersMask, trace,
          "chance must reset pending actions");

  CardMask dealt = 0;
  for (CardId card : cards) {
    const CardMask bit = CardBit(card);
    Require((parent.board.mask & bit) == 0, trace,
            "chance card cannot overlap old board");
    Require((dealt & bit) == 0, trace, "chance cards must be unique");
    dealt |= bit;
  }
}

std::vector<GameAction> AvailableActions(const BettingAbstraction& betting,
                                         const ExactGameState& state) {
  const auto menu = betting.actions_for_betting_node(state.betting);
  return std::vector<GameAction>(menu.actions.begin(),
                                 menu.actions.begin() + menu.count);
}

void TraverseBettingTree(const BettingAbstraction& betting,
                         const ExactGameState& state,
                         int total_chips,
                         int depth,
                         int max_depth,
                         ReachableTrace trace) {
  ValidateState(state, total_chips, trace);
  if (depth >= max_depth || Terminal(state) || BettingRoundOver(state)) {
    return;
  }

  const int player = state.betting.player_to_act;
  Require(IsPlayer(player), trace, "betting tree node must have a player");
  const std::vector<GameAction> actions = AvailableActions(betting, state);
  Require(!actions.empty(), trace, "betting tree node must have actions");

  for (const GameAction& action : actions) {
    ExactGameState child =
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
  ExactGameState state = InitialState(config);
  const int total_chips = TotalChips(state);
  ValidateState(state, total_chips, trace);

  for (int step = 0; step < max_steps && !Terminal(state); ++step) {
    const int player = state.betting.player_to_act;
    if (IsPlayer(player)) {
      const std::vector<GameAction> actions = AvailableActions(betting, state);
      Require(!actions.empty(), trace, "player node must have legal actions");
      for (const GameAction& action : actions) {
        ExactGameState child = state;
        child.betting = ApplyAction(state.betting, action);
        ValidateActionTransition(state, child, player, total_chips, trace);
      }

      std::uniform_int_distribution<size_t> choose(0, actions.size() - 1);
      const GameAction action = actions[choose(rng)];
      const ExactGameState parent = state;
      state.betting = ApplyAction(state.betting, action);
      trace.steps.push_back("action " + ActionString(action) + "\n  before " +
                            StateString(parent) + "\n  after  " +
                            StateString(state));
      ValidateActionTransition(parent, state, player, total_chips, trace);
    } else {
      const ExactGameState parent = state;
      const auto cards = SampleStreetCards(
          state.betting.street, state.board, 0, rng);
      state = ApplyChance(state, cards);
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

ExactGameState FlopState() {
  ExactGameState state;
  state.betting.stack = {18, 18};
  state.betting.street = StreetKind::kFlop;
  state.betting.player_to_act = 1;
  state.betting.folded_player = -1;
  state.betting.committed = {2, 2};
  state.board.add(MakeCardId(2, SuitKind::kHearts));
  state.board.add(MakeCardId(7, SuitKind::kDiamonds));
  state.board.add(MakeCardId(12, SuitKind::kClubs));
  return state;
}

ExactGameState RiverState() {
  ExactGameState state;
  state.betting.stack = {20, 20};
  state.betting.street = StreetKind::kRiver;
  state.betting.player_to_act = 1;
  state.betting.folded_player = -1;
  state.betting.committed = {10, 10};
  state.board.add(MakeCardId(2, SuitKind::kHearts));
  state.board.add(MakeCardId(7, SuitKind::kDiamonds));
  state.board.add(MakeCardId(12, SuitKind::kClubs));
  state.board.add(MakeCardId(9, SuitKind::kSpades));
  state.board.add(MakeCardId(3, SuitKind::kHearts));
  return state;
}

void CheckCompletedRound(const ExactGameState& state, bool terminal) {
  CAPTURE(StateString(state));
  CHECK(BettingRoundOver(state));
  CHECK(state.betting.player_to_act == -1);
  CHECK(state.betting.player_to_act == -1);
  CHECK(Terminal(state) == terminal);
}

ExactGameState RiverShowdown(std::initializer_list<CardId> board) {
  ExactGameState state;
  state.betting.stack = {0, 0};
  state.betting.street = StreetKind::kRiver;
  state.betting.player_to_act = -1;
  state.betting.folded_player = -1;
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
  const int comparison =
      evaluator.compare_hands(a_hand, b_hand, state.board);
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
  CAPTURE(expected_a);
  CAPTURE(StateString(state));
  const double actual_a = GetUtility(state, a_hand, b_hand);
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
  const ExactGameState root = InitialState(config);
  CHECK_THROWS(ApplyAction(root.betting, {ActionKind::kCheck, 0}));
  CHECK_THROWS(ApplyAction(root.betting, {ActionKind::kRaise, 1}));
  CHECK_THROWS(ApplyAction(
      root.betting, {ActionKind::kRaise, root.betting.stack[0]}));

  const ExactGameState flop = FlopState();
  CHECK_THROWS(ApplyAction(flop.betting, {ActionKind::kCall, 0}));
  CHECK_THROWS(ApplyAction(
      flop.betting, {ActionKind::kBet, flop.betting.stack[1]}));

  const ExactGameState folded =
      ApplyStateAction(root, {ActionKind::kFold, 0});
  CHECK_THROWS(ApplyAction(folded.betting, {ActionKind::kCall, 1}));

  ExactGameState broke = flop;
  broke.betting.stack[1] = 0;
  broke.betting.player_to_act = 1;
  CHECK_THROWS(ApplyAction(broke.betting, {ActionKind::kCheck, 0}));
}

TEST_CASE("deterministic action transitions preserve chip accounting") {
  const SolverConfig config = TestConfig();
  ReachableTrace trace{0, 0, config, {}};
  const ExactGameState root = InitialState(config);
  const int total_chips = TotalChips(root);
  ValidateState(root, total_chips, trace);
  CHECK(root.betting.street == StreetKind::kPreflop);
  CHECK(root.betting.player_to_act == 0);
  CHECK(root.betting.committed[0] == config.small_blind);
  CHECK(root.betting.committed[1] == config.big_blind);

  SolverConfig blind_config = config;
  blind_config.starting_stack_size = blind_config.big_blind;
  ReachableTrace blind_trace{0, 0, blind_config, {}};
  ExactGameState blind_root = InitialState(blind_config);
  const int blind_total = TotalChips(blind_root);
  CHECK(blind_root.betting.stack[1] == 0);
  CHECK(AnyPlayerAllIn(blind_root.betting));
  CHECK(!BettingRoundOver(blind_root));
  blind_root = ApplyChecked(
      blind_root, {ActionKind::kCall}, blind_total, blind_trace);
  CHECK(blind_root.betting.stack[0] == 0);
  CHECK(blind_root.betting.stack[1] == 0);
  CheckCompletedRound(blind_root, false);

  ExactGameState call_check =
      ApplyChecked(root, {ActionKind::kCall, 1}, total_chips, trace);
  call_check = ApplyChecked(call_check, {ActionKind::kCheck}, total_chips,
                            trace);
  CheckCompletedRound(call_check, false);

  ExactGameState fold =
      ApplyChecked(root, {ActionKind::kFold}, total_chips, trace);
  CheckCompletedRound(fold, true);

  ExactGameState raise_call =
      ApplyChecked(root, {ActionKind::kRaise, 4}, total_chips, trace);
  raise_call = ApplyChecked(raise_call, {ActionKind::kCall}, total_chips,
                            trace);
  CheckCompletedRound(raise_call, false);

  ExactGameState check_bet_call =
      ApplyChecked(FlopState(), {ActionKind::kCheck}, 40, trace);
  check_bet_call =
      ApplyChecked(check_bet_call, {ActionKind::kBet, 2}, 40, trace);
  check_bet_call =
      ApplyChecked(check_bet_call, {ActionKind::kCall}, 40, trace);
  CheckCompletedRound(check_bet_call, false);

  ExactGameState bet_raise_call =
      ApplyChecked(FlopState(), {ActionKind::kBet, 4}, 40, trace);
  bet_raise_call =
      ApplyChecked(bet_raise_call, {ActionKind::kRaise, 8}, 40, trace);
  bet_raise_call =
      ApplyChecked(bet_raise_call, {ActionKind::kCall}, 40, trace);
  CheckCompletedRound(bet_raise_call, false);

  ExactGameState all_in_raise =
      ApplyChecked(FlopState(), {ActionKind::kBet, 4}, 40, trace);
  all_in_raise =
      ApplyChecked(all_in_raise, {ActionKind::kAllIn}, 40, trace);
  CHECK(all_in_raise.betting.stack[0] == 0);
  CHECK(AnyPlayerAllIn(all_in_raise.betting));
  CHECK(!BettingRoundOver(all_in_raise));
  all_in_raise =
      ApplyChecked(all_in_raise, {ActionKind::kCall}, 40, trace);
  CHECK(all_in_raise.betting.stack[1] == 0);
  CheckCompletedRound(all_in_raise, false);

  ExactGameState all_in_bet =
      ApplyChecked(FlopState(), {ActionKind::kAllIn}, 40, trace);
  CHECK(all_in_bet.betting.stack[1] == 0);
  CHECK(AnyPlayerAllIn(all_in_bet.betting));
  CHECK(!BettingRoundOver(all_in_bet));
  all_in_bet = ApplyChecked(all_in_bet, {ActionKind::kCall}, 40, trace);
  CHECK(all_in_bet.betting.stack[0] == 0);
  CHECK(all_in_bet.betting.stack[1] == 0);
  CheckCompletedRound(all_in_bet, false);

  ExactGameState short_call = root;
  short_call.betting.stack[0] = 3;
  short_call.betting.stack[1] = 12;
  short_call.betting.committed = {1, 8};
  short_call = ApplyChecked(short_call, {ActionKind::kCall}, 24, trace);
  CheckCompletedRound(short_call, false);
  CHECK(short_call.betting.stack[0] == 0);
  CHECK(AnyPlayerAllIn(short_call.betting));
  CHECK(short_call.betting.committed[0] <
        short_call.betting.committed[1]);

  ExactGameState river_check =
      ApplyChecked(RiverState(), {ActionKind::kCheck}, 60, trace);
  river_check = ApplyChecked(river_check, {ActionKind::kCheck}, 60, trace);
  CheckCompletedRound(river_check, true);

  ExactGameState river_call =
      ApplyChecked(RiverState(), {ActionKind::kBet, 10}, 60, trace);
  river_call = ApplyChecked(river_call, {ActionKind::kCall}, 60, trace);
  CheckCompletedRound(river_call, true);

  ExactGameState river_fold =
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

  const ExactGameState root = InitialState(config);
  const int total_chips = TotalChips(root);
  ReachableTrace trace{0, 8, config, {}};
  TraverseBettingTree(BettingAbstraction(config), root, total_chips, 0, 8,
                      trace);
}

TEST_CASE("betting-round completion cases agree") {
  const SolverConfig config = TestConfig();
  const ExactGameState root = InitialState(config);
  ExactGameState preflop_complete =
      ApplyStateAction(root, {ActionKind::kCall, 1});
  CHECK(!BettingRoundOver(preflop_complete));
  preflop_complete =
      ApplyStateAction(preflop_complete, {ActionKind::kCheck});
  CheckCompletedRound(preflop_complete, false);

  ExactGameState checked =
      ApplyStateAction(FlopState(), {ActionKind::kCheck});
  checked = ApplyStateAction(checked, {ActionKind::kCheck});
  CheckCompletedRound(checked, false);

  ExactGameState bet_call =
      ApplyStateAction(FlopState(), {ActionKind::kBet, 4});
  bet_call = ApplyStateAction(bet_call, {ActionKind::kCall, 4});
  CheckCompletedRound(bet_call, false);

  ExactGameState bet_fold =
      ApplyStateAction(FlopState(), {ActionKind::kBet, 4});
  bet_fold = ApplyStateAction(bet_fold, {ActionKind::kFold});
  CheckCompletedRound(bet_fold, true);

  ExactGameState raise_call =
      ApplyStateAction(root, {ActionKind::kRaise, 4});
  raise_call = ApplyStateAction(raise_call, {ActionKind::kCall, 3});
  CheckCompletedRound(raise_call, false);

  ExactGameState short_call = root;
  short_call.betting.stack[0] = 3;
  short_call.betting.stack[1] = 12;
  short_call.betting.committed = {1, 8};
  short_call = ApplyStateAction(short_call, {ActionKind::kCall, 7});
  CheckCompletedRound(short_call, false);
  CHECK(short_call.betting.stack[0] == 0);
  CHECK(short_call.betting.committed[0] <
        short_call.betting.committed[1]);

  ExactGameState all_in_bet =
      ApplyStateAction(FlopState(), {ActionKind::kAllIn});
  all_in_bet = ApplyStateAction(all_in_bet, {ActionKind::kCall});
  CheckCompletedRound(all_in_bet, false);

  ExactGameState runout =
      ApplyStateAction(root, {ActionKind::kCall, 1});
  runout = ApplyStateAction(runout, {ActionKind::kCheck});
  runout = ApplyChance(runout, {MakeCardId(2, SuitKind::kHearts),
                                MakeCardId(7, SuitKind::kDiamonds),
                                MakeCardId(12, SuitKind::kClubs)});
  runout = ApplyStateAction(runout, {ActionKind::kAllIn});
  runout = ApplyStateAction(runout, {ActionKind::kCall});
  runout = ApplyChance(runout, {MakeCardId(9, SuitKind::kSpades)});
  CHECK(runout.betting.player_to_act == -1);
  runout = ApplyChance(runout, {MakeCardId(3, SuitKind::kHearts)});
  CHECK(Terminal(runout));
  CHECK(runout.betting.player_to_act == -1);
  CHECK(runout.betting.player_to_act == -1);
}

TEST_CASE("chance sampling excludes known cards") {
  ExactGameState state = InitialState(TestConfig());
  state = ApplyStateAction(state, {ActionKind::kCall, 1});

  const ComboId a_hand =
      Combo(14, SuitKind::kHearts, 13, SuitKind::kHearts);
  const ComboId b_hand =
      Combo(12, SuitKind::kClubs, 11, SuitKind::kClubs);
  const CardMask blocked = ComboMask(a_hand) | ComboMask(b_hand);
  std::mt19937 rng(123);
  for (int i = 0; i < 500; ++i) {
    const auto cards = SampleStreetCards(
        state.betting.street, state.board, blocked, rng);
    CHECK(cards.size() == 3);
    for (CardId card : cards) {
      CHECK((blocked & CardBit(card)) == 0);
    }
  }
}

TEST_CASE("terminal utility matches independent oracle") {
  ExactGameState folded = InitialState(TestConfig());
  folded = ApplyStateAction(folded, {ActionKind::kFold});
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
