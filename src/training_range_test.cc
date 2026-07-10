#include "src/training_range.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include "src/hand_range.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kHandTypeCount = 169;
using RawRangeEntry = std::array<uint16_t, 2>;

struct RangeOracle {
  std::array<double, kHandTypeCount> hand_weights = {};
  std::array<double, poker::kComboCount> exact_weights = {};
  double total_weight = 0.0;

  void set_hand(int index, double weight) {
    total_weight += weight - hand_weights[static_cast<size_t>(index)];
    hand_weights[static_cast<size_t>(index)] = weight;
  }

  void set_exact(poker::ComboId combo, double weight) {
    total_weight += weight - exact_weights[combo];
    exact_weights[combo] = weight;
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

::poker::ComboId MakeCombo(int first_rank,
                           poker::SuitKind first_suit,
                           int second_rank,
                           poker::SuitKind second_suit) {
  return poker::CardsToComboId(poker::MakeCardId(first_rank, first_suit),
                               poker::MakeCardId(second_rank, second_suit));
}

float TotalWeight(const poker::TrainingRange& range) {
  float total = 0.0f;
  for (uint16_t i = 0; i < range.active_count; ++i) {
    total += range.weights[range.active[i]];
  }
  return total;
}

std::array<int, kHandTypeCount> HandTypeComboCounts() {
  std::array<int, kHandTypeCount> counts = {};
  for (poker::ComboId combo = 0; combo < poker::kComboCount; ++combo) {
    const int index = poker::HandRange::combo_to_index(combo);
    ++counts[static_cast<size_t>(index)];
  }
  return counts;
}

bool Near(double actual, double expected) {
  const double scale = std::max(1.0, std::abs(expected));
  return std::abs(actual - expected) <= 1e-4 * scale;
}

double ExpectedWeight(
    const RangeOracle& oracle,
    poker::ComboId combo,
    const std::array<int, kHandTypeCount>& combo_counts) {
  const int index = poker::HandRange::combo_to_index(combo);
  const double class_weight =
      oracle.hand_weights[static_cast<size_t>(index)] /
      combo_counts[static_cast<size_t>(index)];
  return oracle.exact_weights[combo] + class_weight;
}

void CheckRange(
    const poker::HandRange& hand_range,
    const RangeOracle& oracle,
    const std::array<int, kHandTypeCount>& combo_counts) {
  const poker::TrainingRange range = poker::BuildTrainingRange(hand_range);
  RC_ASSERT(Near(hand_range.get_total_weight(), oracle.total_weight));

  std::array<bool, poker::kComboCount> seen = {};
  for (uint16_t i = 0; i < range.active_count; ++i) {
    const poker::ComboId combo = range.active[i];
    RC_ASSERT(combo < poker::kComboCount);
    RC_ASSERT(range.weights[combo] > 0.0f);
    RC_ASSERT(!seen[combo]);
    seen[combo] = true;
  }

  uint16_t expected_active = 0;
  double expanded_total = 0.0;
  for (poker::ComboId combo = 0; combo < poker::kComboCount; ++combo) {
    const double expected = ExpectedWeight(oracle, combo, combo_counts);
    RC_ASSERT(Near(range.weight(combo), expected));
    const double probability =
        oracle.total_weight > 0.0 ? expected / oracle.total_weight : 0.0;
    RC_ASSERT(Near(hand_range.get_probability(combo), probability));
    if (expected > 0.0) {
      ++expected_active;
    }
    expanded_total += range.weight(combo);
  }
  RC_ASSERT(range.active_count == expected_active);
  RC_ASSERT(Near(expanded_total, oracle.total_weight));
}

poker::TrainingRange SparseRange(
    const std::vector<RawRangeEntry>& entries) {
  poker::TrainingRange range;
  for (const RawRangeEntry& entry : entries) {
    const poker::ComboId combo =
        static_cast<poker::ComboId>(entry[0] % poker::kComboCount);
    range.add(combo, static_cast<float>(1 + entry[1] % 100));
  }
  return range;
}

void CheckFiltered(const poker::TrainingRange& source,
                   poker::CardMask blocked,
                   const poker::TrainingRangeView& filtered) {
  std::array<bool, poker::kComboCount> seen = {};
  for (uint16_t i = 0; i < filtered.active_count; ++i) {
    const poker::ComboId combo = filtered.active[i];
    RC_ASSERT(!seen[combo]);
    seen[combo] = true;
  }

  uint16_t expected_count = 0;
  for (poker::ComboId combo = 0; combo < poker::kComboCount; ++combo) {
    const bool selected =
        source.weights[combo] > 0.0f &&
        (poker::ComboMask(combo) & blocked) == 0;
    const float expected = selected ? source.weights[combo] : 0.0f;
    RC_ASSERT(filtered.weights[combo] == expected);
    if (selected) {
      ++expected_count;
      RC_ASSERT(seen[combo]);
    }
  }
  RC_ASSERT(filtered.active_count == expected_count);
}

bool IsActive(const poker::TrainingRange& range, poker::ComboId combo) {
  for (uint16_t i = 0; i < range.active_count; ++i) {
    if (range.active[i] == combo) {
      return true;
    }
  }
  return false;
}

TEST_CASE("training range preserves hand range expansion") {
  poker::HandRange hand_range;
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AA"), 6.0);
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AKs"), 4.0);
  const poker::ComboId exact_aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  hand_range.add_combo(exact_aces, 2.0);

  const poker::TrainingRange range = poker::BuildTrainingRange(hand_range);
  CHECK(range.active_count == 10);
  CHECK(range.weight(exact_aces) == doctest::Approx(3.0f).epsilon(1e-6));
  CHECK(TotalWeight(range) ==
        doctest::Approx(static_cast<float>(hand_range.get_total_weight()))
            .epsilon(1e-6));

  poker::HandRange empty;
  CHECK(poker::BuildTrainingRange(empty).empty());
}

TEST_CASE("range mutations match an exact-combo oracle") {
  const bool passed = rc::check("range mutation model", [] {
    using RawCommand = std::array<uint16_t, 3>;
    const auto commands =
        *rc::gen::resize(
             48, rc::gen::arbitrary<std::vector<RawCommand>>())
             .as("commands");
    const auto combo_counts = HandTypeComboCounts();
    poker::HandRange range;
    RangeOracle oracle;

    for (const RawCommand& command : commands) {
      const double weight = 1.0 + command[2] % 100;
      switch (command[0] % 3) {
        case 0: {
          const poker::ComboId combo =
              static_cast<poker::ComboId>(command[1] % poker::kComboCount);
          range.add_combo(combo, weight);
          oracle.set_exact(combo, weight);
          break;
        }
        case 1: {
          const int index = command[1] % kHandTypeCount;
          range.add_hand_by_index(index, weight);
          oracle.set_hand(index, weight);
          break;
        }
        case 2:
          range.normalize();
          oracle.normalize();
          break;
      }
      CheckRange(range, oracle, combo_counts);
    }

    poker::HandRange reversed;
    for (int index = kHandTypeCount - 1; index >= 0; --index) {
      const double weight = oracle.hand_weights[static_cast<size_t>(index)];
      if (weight > 0.0) {
        reversed.add_hand_by_index(index, weight);
      }
    }
    for (int combo = poker::kComboCount - 1; combo >= 0; --combo) {
      const double weight = oracle.exact_weights[static_cast<size_t>(combo)];
      if (weight > 0.0) {
        reversed.add_combo(static_cast<poker::ComboId>(combo), weight);
      }
    }
    CheckRange(reversed, oracle, combo_counts);
    RC_ASSERT(range.to_string() == reversed.to_string());
  });
  CHECK(passed);
}

TEST_CASE("training range view reuses scratch without stale weights") {
  const bool passed = rc::check("range view filtering", [] {
    const auto entries =
        *rc::gen::resize(
             32, rc::gen::arbitrary<std::vector<RawRangeEntry>>())
             .as("range");
    const poker::CardMask mask_a =
        *rc::gen::arbitrary<poker::CardMask>().as("mask A");
    const poker::CardMask mask_b =
        *rc::gen::arbitrary<poker::CardMask>().as("mask B");
    const poker::TrainingRange source = SparseRange(entries);
    const poker::TrainingRange original = source;
    const poker::TrainingRangeView view(source);
    poker::TrainingRangeView scratch;

    view.copy_without_mask_into(mask_a, scratch);
    CheckFiltered(source, mask_a, scratch);
    const auto first_weights = scratch.weights;

    view.copy_without_mask_into(mask_b, scratch);
    CheckFiltered(source, mask_b, scratch);
    for (poker::ComboId combo = 0; combo < poker::kComboCount; ++combo) {
      if (first_weights[combo] > 0.0f &&
          (poker::ComboMask(combo) & mask_b) != 0) {
        RC_ASSERT(scratch.weights[combo] == 0.0f);
      }
    }

    RC_ASSERT(source.active_count == original.active_count);
    RC_ASSERT(source.active == original.active);
    RC_ASSERT(source.weights == original.weights);
  });
  CHECK(passed);
}

TEST_CASE("range sampler returns active compatible hands") {
  const bool passed = rc::check("compatible range sampling", [] {
    const auto a_entries =
        *rc::gen::resize(
             32, rc::gen::arbitrary<std::vector<RawRangeEntry>>())
             .as("range A");
    const auto b_entries =
        *rc::gen::resize(
             32, rc::gen::arbitrary<std::vector<RawRangeEntry>>())
             .as("range B");
    const uint32_t seed = *rc::gen::arbitrary<uint32_t>().as("seed");
    poker::TrainingRange a_range = SparseRange(a_entries);
    poker::TrainingRange b_range = SparseRange(b_entries);
    const poker::ComboId guaranteed_a = poker::CardsToComboId(0, 1);
    const poker::ComboId guaranteed_b = poker::CardsToComboId(2, 3);
    a_range.add(guaranteed_a, 1.0f);
    b_range.add(guaranteed_b, 1.0f);

    const poker::RangeSampler sampler(a_range, b_range);
    std::mt19937 rng(seed);
    for (int i = 0; i < 64; ++i) {
      const poker::RangeDeal deal = sampler.sample(rng);
      RC_ASSERT(IsActive(a_range, deal.player_a_combo));
      RC_ASSERT(IsActive(b_range, deal.player_b_combo));
      RC_ASSERT(a_range.weight(deal.player_a_combo) > 0.0f);
      RC_ASSERT(b_range.weight(deal.player_b_combo) > 0.0f);
      RC_ASSERT((poker::ComboMask(deal.player_a_combo) &
                 poker::ComboMask(deal.player_b_combo)) == 0);
    }
  });
  CHECK(passed);
}

TEST_CASE("range sampler rejects empty and incompatible ranges") {
  poker::TrainingRange empty;
  poker::TrainingRange nonempty;
  nonempty.add(poker::CardsToComboId(0, 1), 1.0f);
  CHECK_THROWS_AS(poker::RangeSampler(empty, nonempty),
                  std::invalid_argument);

  poker::TrainingRange a_range;
  a_range.add(poker::CardsToComboId(0, 1), 1.0f);
  poker::TrainingRange b_range;
  b_range.add(poker::CardsToComboId(0, 2), 1.0f);
  b_range.add(poker::CardsToComboId(1, 3), 1.0f);
  CHECK_THROWS_AS(poker::RangeSampler(a_range, b_range),
                  std::invalid_argument);
}

}  // namespace
