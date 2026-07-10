#include "src/hand_range.h"
#include "src/training_range.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace poker {
namespace {

constexpr int kHandTypeCount = 169;
using RawRangeEntry = std::array<uint16_t, 2>;

ComboId Combo(int first_rank,
              SuitKind first_suit,
              int second_rank,
              SuitKind second_suit) {
  return CardsToComboId(MakeCardId(first_rank, first_suit),
                        MakeCardId(second_rank, second_suit));
}

struct RangeOracle {
  std::array<double, kHandTypeCount> hand_weights = {};
  std::array<double, kComboCount> exact_weights = {};
  double total_weight = 0.0;

  void set(double& slot, double weight) {
    total_weight += weight - slot;
    slot = weight;
  }

  void normalize() {
    if (total_weight <= 0.0) {
      return;
    }
    for (double& weight : hand_weights) {
      weight /= total_weight;
    }
    for (double& weight : exact_weights) {
      weight /= total_weight;
    }
    total_weight = 1.0;
  }
};

float TotalWeight(const TrainingRange& range) {
  float total = 0.0f;
  for (uint16_t i = 0; i < range.active_count; ++i) {
    total += range.weights[range.active[i]];
  }
  return total;
}

std::array<int, kHandTypeCount> HandTypeComboCounts() {
  std::array<int, kHandTypeCount> counts = {};
  for (ComboId combo = 0; combo < kComboCount; ++combo) {
    ++counts[HandRange::combo_to_index(combo)];
  }
  return counts;
}

bool Near(double actual, double expected) {
  return std::abs(actual - expected) <=
         1e-4 * std::max(1.0, std::abs(expected));
}

double ExpectedWeight(
    const RangeOracle& oracle,
    ComboId combo,
    const std::array<int, kHandTypeCount>& combo_counts) {
  const int index = HandRange::combo_to_index(combo);
  return oracle.exact_weights[combo] +
         oracle.hand_weights[index] / combo_counts[index];
}

void CheckRange(
    const HandRange& hand_range,
    const RangeOracle& oracle,
    const std::array<int, kHandTypeCount>& combo_counts) {
  const TrainingRange range = BuildTrainingRange(hand_range);
  RC_ASSERT(Near(hand_range.get_total_weight(), oracle.total_weight));

  std::array<bool, kComboCount> seen = {};
  for (uint16_t i = 0; i < range.active_count; ++i) {
    const ComboId combo = range.active[i];
    RC_ASSERT(combo < kComboCount);
    RC_ASSERT(range.weights[combo] > 0.0f);
    RC_ASSERT(!seen[combo]);
    seen[combo] = true;
  }

  uint16_t expected_active = 0;
  double expanded_total = 0.0;
  for (ComboId combo = 0; combo < kComboCount; ++combo) {
    const double expected = ExpectedWeight(oracle, combo, combo_counts);
    RC_ASSERT(Near(range.weight(combo), expected));
    const double probability =
        oracle.total_weight > 0.0 ? expected / oracle.total_weight : 0.0;
    RC_ASSERT(Near(hand_range.get_probability(combo), probability));
    expected_active += expected > 0.0;
    expanded_total += range.weight(combo);
  }
  RC_ASSERT(range.active_count == expected_active);
  RC_ASSERT(Near(expanded_total, oracle.total_weight));
}

TrainingRange SparseRange(const std::vector<RawRangeEntry>& entries) {
  TrainingRange range;
  for (const RawRangeEntry& entry : entries) {
    range.add(static_cast<ComboId>(entry[0] % kComboCount),
              static_cast<float>(1 + entry[1] % 100));
  }
  return range;
}

void CheckFiltered(const TrainingRange& source,
                   CardMask blocked,
                   const TrainingRangeView& filtered) {
  std::array<bool, kComboCount> seen = {};
  for (uint16_t i = 0; i < filtered.active_count; ++i) {
    const ComboId combo = filtered.active[i];
    RC_ASSERT(!seen[combo]);
    seen[combo] = true;
  }

  uint16_t expected_count = 0;
  for (ComboId combo = 0; combo < kComboCount; ++combo) {
    const bool selected =
        source.weights[combo] > 0.0f &&
        (ComboMask(combo) & blocked) == 0;
    RC_ASSERT(filtered.weights[combo] ==
              (selected ? source.weights[combo] : 0.0f));
    if (selected) {
      ++expected_count;
      RC_ASSERT(seen[combo]);
    }
  }
  RC_ASSERT(filtered.active_count == expected_count);
}

bool IsActive(const TrainingRange& range, ComboId combo) {
  for (uint16_t i = 0; i < range.active_count; ++i) {
    if (range.active[i] == combo) {
      return true;
    }
  }
  return false;
}

TEST_CASE("range syntax and hand-type indexing are canonical") {
  struct ParserCase {
    const char* text;
    uint16_t active_count;
  };
  const ParserCase cases[] = {
      {"AA", 6},   {"AKs", 4}, {"AKo", 12}, {"AK", 16},
      {"AA,KK", 12}, {"QQ+", 18}, {"89s+", 0},
  };
  for (const ParserCase& test : cases) {
    CAPTURE(test.text);
    HandRange range;
    range.set_from_string(test.text);
    CHECK(BuildTrainingRange(range).active_count == test.active_count);
  }

  for (int index = 0; index < kHandTypeCount; ++index) {
    CAPTURE(index);
    const std::optional<ComboId> combo = HandRange::index_to_combo(index);
    REQUIRE(combo.has_value());
    CHECK(HandRange::combo_to_index(*combo) == index);
  }
  CHECK(!HandRange::index_to_combo(-1).has_value());
  CHECK(!HandRange::index_to_combo(kHandTypeCount).has_value());
}

TEST_CASE("training ranges combine class and exact-combo weights") {
  HandRange hand_range;
  hand_range.add_hand_by_index(HandRange::string_to_index("AA"), 6.0);
  hand_range.add_hand_by_index(HandRange::string_to_index("AKs"), 4.0);
  const ComboId exact_aces =
      Combo(14, SuitKind::kSpades, 14, SuitKind::kHearts);
  hand_range.add_combo(exact_aces, 2.0);

  const TrainingRange range = BuildTrainingRange(hand_range);
  CHECK(range.active_count == 10);
  CHECK(range.weight(exact_aces) == doctest::Approx(3.0f).epsilon(1e-6));
  CHECK(TotalWeight(range) ==
        doctest::Approx(static_cast<float>(hand_range.get_total_weight()))
            .epsilon(1e-6));
  CHECK(BuildTrainingRange(HandRange{}).empty());
}

TEST_CASE("range mutations match an exact-combo oracle") {
  const bool passed = rc::check("range mutation model", [] {
    using RawCommand = std::array<uint16_t, 3>;
    const auto commands =
        *rc::gen::resize(48,
                         rc::gen::arbitrary<std::vector<RawCommand>>())
             .as("commands");
    const auto combo_counts = HandTypeComboCounts();
    HandRange range;
    RangeOracle oracle;

    for (const RawCommand& command : commands) {
      const double weight = 1.0 + command[2] % 100;
      switch (command[0] % 3) {
        case 0: {
          const ComboId combo =
              static_cast<ComboId>(command[1] % kComboCount);
          range.add_combo(combo, weight);
          oracle.set(oracle.exact_weights[combo], weight);
          break;
        }
        case 1: {
          const int index = command[1] % kHandTypeCount;
          range.add_hand_by_index(index, weight);
          oracle.set(oracle.hand_weights[index], weight);
          break;
        }
        case 2:
          range.normalize();
          oracle.normalize();
          break;
      }
      CheckRange(range, oracle, combo_counts);
    }

    HandRange reversed;
    for (int index = kHandTypeCount - 1; index >= 0; --index) {
      if (oracle.hand_weights[index] > 0.0) {
        reversed.add_hand_by_index(index, oracle.hand_weights[index]);
      }
    }
    for (int combo = kComboCount - 1; combo >= 0; --combo) {
      if (oracle.exact_weights[combo] > 0.0) {
        reversed.add_combo(static_cast<ComboId>(combo),
                           oracle.exact_weights[combo]);
      }
    }
    CheckRange(reversed, oracle, combo_counts);
    RC_ASSERT(range.to_string() == reversed.to_string());
  });
  CHECK(passed);
}

TEST_CASE("range views reuse scratch without stale weights") {
  const bool passed = rc::check("range view filtering", [] {
    const auto entries =
        *rc::gen::resize(32,
                         rc::gen::arbitrary<std::vector<RawRangeEntry>>())
             .as("range");
    const CardMask mask_a =
        *rc::gen::arbitrary<CardMask>().as("mask A");
    const CardMask mask_b =
        *rc::gen::arbitrary<CardMask>().as("mask B");
    const TrainingRange source = SparseRange(entries);
    const TrainingRange original = source;
    const TrainingRangeView view(source);
    TrainingRangeView scratch;

    view.copy_without_mask_into(mask_a, scratch);
    CheckFiltered(source, mask_a, scratch);
    const auto first_weights = scratch.weights;

    view.copy_without_mask_into(mask_b, scratch);
    CheckFiltered(source, mask_b, scratch);
    for (ComboId combo = 0; combo < kComboCount; ++combo) {
      if (first_weights[combo] > 0.0f &&
          (ComboMask(combo) & mask_b) != 0) {
        RC_ASSERT(scratch.weights[combo] == 0.0f);
      }
    }

    RC_ASSERT(source.active_count == original.active_count);
    RC_ASSERT(source.active == original.active);
    RC_ASSERT(source.weights == original.weights);
  });
  CHECK(passed);
}

TEST_CASE("range sampling returns active compatible hands") {
  const bool passed = rc::check("compatible range sampling", [] {
    const auto a_entries =
        *rc::gen::resize(32,
                         rc::gen::arbitrary<std::vector<RawRangeEntry>>())
             .as("range A");
    const auto b_entries =
        *rc::gen::resize(32,
                         rc::gen::arbitrary<std::vector<RawRangeEntry>>())
             .as("range B");
    const uint32_t seed = *rc::gen::arbitrary<uint32_t>().as("seed");
    TrainingRange a_range = SparseRange(a_entries);
    TrainingRange b_range = SparseRange(b_entries);
    a_range.add(CardsToComboId(0, 1), 1.0f);
    b_range.add(CardsToComboId(2, 3), 1.0f);

    const RangeSampler sampler(a_range, b_range);
    std::mt19937 rng(seed);
    for (int i = 0; i < 64; ++i) {
      const RangeDeal deal = sampler.sample(rng);
      RC_ASSERT(IsActive(a_range, deal.player_a_combo));
      RC_ASSERT(IsActive(b_range, deal.player_b_combo));
      RC_ASSERT(a_range.weight(deal.player_a_combo) > 0.0f);
      RC_ASSERT(b_range.weight(deal.player_b_combo) > 0.0f);
      RC_ASSERT((ComboMask(deal.player_a_combo) &
                 ComboMask(deal.player_b_combo)) == 0);
    }
  });
  CHECK(passed);
}

TEST_CASE("range sampling rejects empty or incompatible inputs") {
  TrainingRange empty;
  TrainingRange nonempty;
  nonempty.add(CardsToComboId(0, 1), 1.0f);
  CHECK_THROWS_AS(RangeSampler(empty, nonempty), std::invalid_argument);

  TrainingRange a_range;
  a_range.add(CardsToComboId(0, 1), 1.0f);
  TrainingRange b_range;
  b_range.add(CardsToComboId(0, 2), 1.0f);
  b_range.add(CardsToComboId(1, 3), 1.0f);
  CHECK_THROWS_AS(RangeSampler(a_range, b_range), std::invalid_argument);
}

}  // namespace
}  // namespace poker
