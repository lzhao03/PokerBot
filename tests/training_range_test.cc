#include "src/solver.h"

#include <cmath>
#include <stdexcept>

#include "doctest/doctest.h"

namespace poker {
namespace {

using S = SuitKind;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(MakeCardId(r0, s0), MakeCardId(r1, s1));
}

TEST_CASE("range syntax expands to exact combo weights") {
  struct Case {
    const char* text;
    size_t count;
  };
  const Case cases[] = {
      {"AA", 6}, {"AKs", 4}, {"AKo", 12}, {"AK", 16},
      {"AA,KK", 12}, {"QQ+", 18}, {"89s+", 0}};
  for (const Case& test : cases) {
    CAPTURE(test.text);
    CHECK(ParseRange(test.text).count() == test.count);
  }

  const ComboRange aces = ParseRange("AA");
  const ComboId hand = H(14, S::kSpades, 14, S::kHearts);
  CHECK(aces.weight(hand) == doctest::Approx(1.0f / 6.0f));
  CHECK(UniformRange().count() == kComboCount);
  CHECK(SingleComboRange(hand, 2.0f).weight(hand) == 2.0f);
}

TEST_CASE("deal sampling rejects incompatible ranges") {
  SolverConfig config;
  config.starting_stack = 8;
  for (auto& sizes : config.bet_sizes) {
    sizes = {1.0};
  }

  const ComboRange a = SingleComboRange(CardsToComboId(kDeck[0], kDeck[1]));
  ComboRange b;
  b.add(CardsToComboId(kDeck[0], kDeck[2]));
  b.add(CardsToComboId(kDeck[1], kDeck[3]));
  CFRSolver solver(config);
  CHECK_THROWS_AS(solver.run(1, a, b), std::invalid_argument);

  const ComboRange compatible =
      SingleComboRange(H(12, S::kClubs, 12, S::kDiamonds));
  solver.run(1, a, compatible);
  CHECK(std::isfinite(solver.get_expected_value(0)));
}

}  // namespace
}  // namespace poker
