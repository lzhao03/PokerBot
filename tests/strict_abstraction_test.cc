#include "src/solver.h"

#include "doctest/doctest.h"

#include <array>
#include <cmath>

namespace poker {
namespace {

Card C(int rank, Suit suit) {
  return Card(static_cast<Rank>(rank - 2), suit);
}

ComboRange Range(int first_rank, int second_rank, Suit suit) {
  const ComboId combo = CardsToComboId(
      C(first_rank, suit), C(second_rank, suit));
  ComboRange range;
  range.add(combo);
  return range;
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
    SolverConfig config;
    config.card_abstraction = abstraction;
    for (auto& fractions : config.bet_abstraction.pot_fractions) {
      fractions = {1.0};
    }
    config.max_info_sets = 500000;
    config.starting_stacks = {8, 8};

    auto solver = TabularCfrSolver::Create(
        config, {Range(14, 13, Suit::Hearts), Range(12, 11, Suit::Clubs)});
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
