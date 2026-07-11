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
  CHECK(aces.weight(hand) == 1.0f);

  const ComboRange uniform = UniformComboRange();
  CHECK(uniform.count() == kComboCount);
  for (ComboId combo : uniform.active) {
    CHECK(uniform.weight(combo) == 1.0f);
  }

  const auto suited = ParseRange("AKs");
  const auto offsuit = ParseRange("AKo");
  const auto any = ParseRange("AK");
  const auto split = ParseRange("AKs,AKo");
  REQUIRE(suited.ok());
  REQUIRE(offsuit.ok());
  REQUIRE(any.ok());
  REQUIRE(split.ok());
  CHECK(suited->count() == 4);
  CHECK(offsuit->count() == 12);
  CHECK(any->count() == 16);
  CHECK(split->weights == any->weights);
  CHECK(SingleComboRange(hand, 2.0f).weight(hand) == 2.0f);
  CHECK_FALSE(ParseRange("89s+").ok());
  CHECK_FALSE(ParseRange("AA,,KK").ok());
  CHECK_FALSE(ParseRange("").ok());
}

TEST_CASE("deal sampling rejects incompatible ranges") {
  SolverConfigOptions options;
  options.starting_stack = 8;
  for (auto& fractions : options.bet_abstraction.pot_fractions) {
    fractions = {1.0};
  }
  const auto config_result = SolverConfig::Create(std::move(options));
  REQUIRE(config_result.ok());
  const SolverConfig config = *config_result;

  const ComboRange a = SingleComboRange(CardsToComboId(kDeck[0], kDeck[1]));
  ComboRange b;
  b.add(CardsToComboId(kDeck[0], kDeck[2]));
  b.add(CardsToComboId(kDeck[1], kDeck[3]));
  const Chips stack = config.starting_stack();
  const ExactPublicState root = MakeInitialState(
      BettingRules{config.big_blind()}, {stack, stack},
      {config.small_blind(), config.big_blind()});
  CHECK_FALSE(CFRSolver::Create({config, root, {a, b}}).ok());

  const ComboRange compatible =
      SingleComboRange(H(12, S::kClubs, 12, S::kDiamonds));
  auto solver = CFRSolver::Create({config, root, {a, compatible}});
  REQUIRE(solver.ok());
  (*solver)->run(1);
  CHECK(std::isfinite((*solver)->get_expected_value(Player::kA)));
}

}  // namespace
}  // namespace poker
