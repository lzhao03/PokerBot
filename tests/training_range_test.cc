#include "src/solver.h"

#include <cmath>
#include <stdexcept>

#include "doctest/doctest.h"

namespace poker {
namespace {

using S = Suit;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(Card(static_cast<Rank>(r0 - 2), s0),
                        Card(static_cast<Rank>(r1 - 2), s1));
}

TEST_CASE("range syntax expands to exact combo weights") {
  struct Case {
    const char* text;
    size_t count;
  };
  const Case cases[] = {
      {"AA", 6}, {"AKs", 4}, {"AKo", 12}, {"AK", 16},
      {"AA,KK", 12}, {"QQ+", 18}};
  for (const Case& test : cases) {
    CAPTURE(test.text);
    const auto range = ParseRange(test.text);
    REQUIRE(range.ok());
    CHECK(range->count() == test.count);
  }

  const auto parsed_aces = ParseRange("AA");
  REQUIRE(parsed_aces.ok());
  const ComboRange aces = *parsed_aces;
  const ComboId hand = H(14, S::kSpades, 14, S::kHearts);
  CHECK(aces.weight(hand) == doctest::Approx(1.0f / 6.0f));
  CHECK(UniformRange().count() == kComboCount);
  CHECK(SingleComboRange(hand, 2.0f).weight(hand) == 2.0f);
  CHECK_FALSE(ParseRange("89s+").ok());
  CHECK_FALSE(ParseRange("AA,,KK").ok());
  CHECK_FALSE(ParseRange("").ok());
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
  CHECK(std::isfinite(solver.get_expected_value(Player::kA)));
}

}  // namespace
}  // namespace poker
