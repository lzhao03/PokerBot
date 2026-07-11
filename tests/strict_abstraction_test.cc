#include "src/solver.h"

#include "doctest/doctest.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace poker {
namespace {

Card C(int rank, Suit suit) {
  return Card(static_cast<Rank>(rank - 2), suit);
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

ComboRange Range(int first_rank, int second_rank, Suit suit) {
  const ComboId combo = CardsToComboId(
      C(first_rank, suit), C(second_rank, suit));
  return SingleComboRange(combo);
}

TEST_CASE("all card abstraction combinations support history traversal") {
  const std::array<CardAbstractionConfig, 4> abstractions = {{
      {PublicCardMode::kExactCanonical, PrivateCardMode::kExactCanonical},
      {PublicCardMode::kExactCanonical, PrivateCardMode::kCoarse},
      {PublicCardMode::kTexture, PrivateCardMode::kExactCanonical},
      {PublicCardMode::kTexture, PrivateCardMode::kCoarse},
  }};
  for (const CardAbstractionConfig& abstraction : abstractions) {
    CAPTURE(static_cast<int>(abstraction.public_mode));
    CAPTURE(static_cast<int>(abstraction.private_mode));
    SolverConfigOptions options;
    options.card_abstraction = abstraction;
    options.starting_stack = 8;
    options.small_blind = 1;
    options.big_blind = 2;
    for (auto& fractions : options.bet_abstraction.pot_fractions) {
      fractions = {1.0};
    }
    options.max_info_sets = 500000;
    options.accumulate_average_strategy = false;
    const auto config_result = SolverConfig::Create(std::move(options));
    REQUIRE(config_result.ok());
    const SolverConfig config = *config_result;

    const BettingRules rules{config.big_blind()};
    ExactPublicState state = MakeInitialState(rules, {8, 8}, {1, 2});
    state.betting = Apply(state.betting, {ActionKind::kCall, 2});
    state.betting = Apply(state.betting, {ActionKind::kCheck, 0});
    const std::array<Card, 3> flop = {
        C(2, Suit::kDiamonds), C(7, Suit::kSpades),
        C(9, Suit::kDiamonds),
    };
    state = DealChance(state, flop, rules);

    CFRSolver solver(config, state);
    const auto deals = DealDistribution::Create(
        Range(14, 13, Suit::kHearts), Range(12, 11, Suit::kClubs));
    REQUIRE(deals.ok());
    solver.run(2, *deals);

    CHECK(solver.get_iterations_run() == 2);
    CHECK(std::isfinite(solver.get_expected_value(Player::kA)));
    CHECK(solver.get_history_count() > 0);
    CHECK(solver.get_info_set_count() > 0);
  }
}

}  // namespace
}  // namespace poker
