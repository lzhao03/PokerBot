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
      {PublicCardMode::ExactCanonical,
       PrivateAbstractionKind::ExactCanonical},
      {PublicCardMode::ExactCanonical,
       PrivateAbstractionKind::Handcrafted36},
      {PublicCardMode::Texture,
       PrivateAbstractionKind::ExactCanonical},
      {PublicCardMode::Texture,
       PrivateAbstractionKind::Handcrafted36},
  }};
  for (const CardAbstractionConfig& abstraction : abstractions) {
    CAPTURE(static_cast<int>(abstraction.public_mode));
    CAPTURE(static_cast<int>(abstraction.private_kind));
    SolverConfig options;
    options.card_abstraction = abstraction;
    for (auto& fractions : options.bet_abstraction.pot_fractions) {
      fractions = {1.0};
    }
    options.max_info_sets = 500000;
    options.accumulate_average_strategy = false;
    const auto config_result = SolverConfig::Create(std::move(options));
    REQUIRE(config_result.ok());
    const SolverConfig config = *config_result;

    const BettingRules& rules = config.betting_rules;
    ExactPublicState state = MakeInitialState(rules, {8, 8}, {1, 2});
    state.betting = Apply(state.betting, {ActionKind::Call, 2});
    state.betting = Apply(state.betting, {ActionKind::Check, 0});
    const std::array<Card, 3> flop = {
        C(2, Suit::Diamonds), C(7, Suit::Spades),
        C(9, Suit::Diamonds),
    };
    state = DealChance(state, flop, rules);

    auto solver = TabularCfrSolver::Create(
        {config, state,
         {Range(14, 13, Suit::Hearts),
          Range(12, 11, Suit::Clubs)}});
    REQUIRE(solver.ok());
    solver->run(2);

    CHECK(solver->iterations() == 2);
    CHECK(std::isfinite(solver->expected_value(Player::A)));
    CHECK(solver->history_count() > 0);
    CHECK(solver->info_set_count() > 0);
  }
}

}  // namespace
}  // namespace poker
