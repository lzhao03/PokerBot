#include "src/hand_range.h"
#include "src/training_range.h"

#include "doctest/doctest.h"

#include <random>
#include <stdexcept>

namespace poker {
namespace {

using S = SuitKind;

ComboId H(int r0, S s0, int r1, S s1) {
  return CardsToComboId(MakeCardId(r0, s0), MakeCardId(r1, s1));
}

TEST_CASE("range syntax and hand classes expand canonically") {
  struct Case { const char* text; uint16_t count; };
  const Case cases[] = {
      {"AA", 6}, {"AKs", 4}, {"AKo", 12}, {"AK", 16},
      {"AA,KK", 12}, {"QQ+", 18}, {"89s+", 0}};
  for (const Case& test : cases) {
    HandRange range;
    range.set_from_string(test.text);
    CAPTURE(test.text);
    CHECK(BuildTrainingRange(range).active_count == test.count);
  }

  for (int index = 0; index < 169; ++index) {
    const auto combo = HandRange::index_to_combo(index);
    REQUIRE(combo.has_value());
    CHECK(HandRange::combo_to_index(*combo) == index);
  }
  CHECK_FALSE(HandRange::index_to_combo(-1).has_value());
  CHECK_FALSE(HandRange::index_to_combo(169).has_value());
}

TEST_CASE("expansion and scratch filtering preserve exact weights") {
  const ComboId aces = H(14, S::kSpades, 14, S::kHearts);
  const ComboId kings = H(13, S::kSpades, 13, S::kHearts);
  const ComboId queens = H(12, S::kSpades, 12, S::kHearts);

  HandRange hand_range;
  hand_range.add_hand_by_index(HandRange::string_to_index("AA"), 6.0);
  hand_range.add_hand_by_index(HandRange::string_to_index("AKs"), 4.0);
  hand_range.add_combo(aces, 2.0);
  const TrainingRange expanded = BuildTrainingRange(hand_range);
  CHECK(expanded.active_count == 10);
  CHECK(expanded.weight(aces) == doctest::Approx(3.0f));

  TrainingRange source;
  source.add(aces, 1.0f);
  source.add(kings, 2.0f);
  source.add(queens, 3.0f);
  const TrainingRange original = source;
  TrainingRangeView view(source);
  TrainingRangeView scratch;
  view.copy_without_mask_into(CardBit(MakeCardId(14, S::kHearts)), scratch);
  CHECK(scratch.weights[aces] == 0.0f);
  CHECK(scratch.weights[kings] == 2.0f);
  view.copy_without_mask_into(CardBit(MakeCardId(13, S::kHearts)), scratch);
  CHECK(scratch.weights[aces] == 1.0f);
  CHECK(scratch.weights[kings] == 0.0f);
  CHECK(scratch.weights[queens] == 3.0f);
  CHECK(source.active == original.active);
  CHECK(source.weights == original.weights);
}

TEST_CASE("range sampling returns active compatible deals") {
  TrainingRange a;
  TrainingRange b;
  a.add(H(14, S::kHearts, 14, S::kSpades), 1.0f);
  a.add(H(13, S::kHearts, 13, S::kSpades), 2.0f);
  b.add(H(12, S::kClubs, 12, S::kDiamonds), 1.0f);
  b.add(H(11, S::kClubs, 11, S::kDiamonds), 2.0f);
  RangeSampler sampler(a, b);
  std::mt19937 rng(7);
  for (int i = 0; i < 64; ++i) {
    const RangeDeal deal = sampler.sample(rng);
    CHECK(a.weight(deal.player_a_combo) > 0.0f);
    CHECK(b.weight(deal.player_b_combo) > 0.0f);
    CHECK((ComboMask(deal.player_a_combo) & ComboMask(deal.player_b_combo)) == 0);
  }

  TrainingRange empty;
  CHECK_THROWS_AS(RangeSampler(empty, a), std::invalid_argument);
  TrainingRange x;
  TrainingRange y;
  x.add(CardsToComboId(0, 1), 1.0f);
  y.add(CardsToComboId(0, 2), 1.0f);
  y.add(CardsToComboId(1, 3), 1.0f);
  CHECK_THROWS_AS(RangeSampler(x, y), std::invalid_argument);
}

}  // namespace
}  // namespace poker
