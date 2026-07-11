#include "src/solver.h"

#include "doctest/doctest.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace poker {
namespace {

#if POKER_COARSE_PUBLIC_BUCKETS == POKER_COARSE_PRIVATE_BUCKETS
#error "strict_abstraction_test requires one coarse abstraction"
#endif

Card C(int rank, Suit suit) {
  return Card(static_cast<Rank>(rank - 2), suit);
}

BettingState Apply(const BettingState& state, GameAction action) {
  const auto* decision = std::get_if<DecisionState>(&state);
  if (decision == nullptr) {
    throw std::invalid_argument("expected decision state");
  }
  const auto child = TryApplyAction(*decision, action);
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

ComboRange Range(int first_rank, int second_rank, Suit suit) {
  const ComboId combo = CardsToComboId(
      C(first_rank, suit), C(second_rank, suit));
  return SingleComboRange(combo);
}

TEST_CASE("mixed abstractions support history traversal") {
  SolverConfig config;
  config.starting_stack = 8;
  config.small_blind = 1;
  config.big_blind = 2;
  for (auto& sizes : config.bet_sizes) {
    sizes = {1.0};
  }
  config.max_info_sets = 500000;
  config.accumulate_average_strategy = false;

  const BettingRules rules{config.big_blind};
  ExactPublicState state = MakeInitialState(rules, {8, 8}, {1, 2});
  state.betting = Apply(
      state.betting, {ActionKind::kCall, 2});
  state.betting = Apply(
      state.betting, {ActionKind::kCheck, 0});
  const std::array<Card, 3> flop = {
      C(2, Suit::kDiamonds), C(7, Suit::kSpades),
      C(9, Suit::kDiamonds),
  };
  state = DealChance(state, flop, rules);

  CFRSolver solver(config, state);
  solver.run(2, Range(14, 13, Suit::kHearts),
             Range(12, 11, Suit::kClubs));

  CHECK(solver.get_iterations_run() == 2);
  CHECK(std::isfinite(solver.get_expected_value(Player::kA)));
  CHECK(solver.get_history_count() > 0);
  CHECK(solver.get_info_set_count() > 0);
}

}  // namespace
}  // namespace poker
