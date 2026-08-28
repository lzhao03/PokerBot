#include "src/solver.h"

#include <cmath>
#include "doctest/doctest.h"

namespace poker {
namespace {

using S = Suit;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(Card(static_cast<Rank>(r0 - 2), s0),
                        Card(static_cast<Rank>(r1 - 2), s1));
}

TEST_CASE("uniform combo range includes every combo equally") {
  const ComboRange uniform = UniformComboRange();
  CHECK(uniform.count() == kComboCount);
  for (float weight : uniform.weights) {
    CHECK(weight == 1.0f);
  }
}

TEST_CASE("deal sampling rejects incompatible ranges") {
  SolverConfig config;
  for (auto& fractions : config.bet_abstraction.pot_fractions) {
    fractions = {1.0};
  }

  ComboRange a;
  a.add(CardsToComboId(kDeck[0], kDeck[1]));
  ComboRange b;
  b.add(CardsToComboId(kDeck[0], kDeck[2]));
  b.add(CardsToComboId(kDeck[1], kDeck[3]));
  const ExactPublicState root = MakeInitialState(
      config.betting_rules, {8, 8}, {1, 2});
  CHECK_FALSE(TabularCfrSolver::Create({config, root, {a, b}}).ok());

  ComboRange compatible;
  compatible.add(H(12, S::Clubs, 12, S::Diamonds));
  auto solver = TabularCfrSolver::Create({config, root, {a, compatible}});
  REQUIRE(solver.ok());
  solver->run(1);
  CHECK(std::isfinite(solver->expected_value(Player::A)));
}

}  // namespace
}  // namespace poker
