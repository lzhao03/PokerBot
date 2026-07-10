#include "src/training_range.h"

#include "doctest/doctest.h"
#include "rapidcheck.h"

#include "src/hand_range.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kHandTypeCount = 169;

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

TEST_CASE("training range view can reset and reuse filtered storage") {
  const poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);
  poker::TrainingRange range;
  range.add(aces, 1.0f);
  range.add(kings, 2.0f);

  poker::TrainingRangeView view(range);
  CHECK(view.size() == 2);

  view.reset_to_filtered();
  view.add(aces, 0.25f);
  view.add(kings, 0.75f);
  CHECK(view.size() == 2);

  view.reset_to_filtered();
  CHECK(view.empty());
  CHECK(view.weights[aces] == 0.0f);
  CHECK(view.weights[kings] == 0.0f);

  view.add(kings, 0.5f);
  CHECK(view.size() == 1);
  CHECK(view.combo(0) == kings);
  CHECK(view.weight(0) == doctest::Approx(0.5f).epsilon(1e-6));
}

TEST_CASE("training range view filters blocked cards into scratch") {
  const poker::CardId ace_spades =
      poker::MakeCardId(14, poker::SuitKind::kSpades);
  const poker::CardId ace_hearts =
      poker::MakeCardId(14, poker::SuitKind::kHearts);
  const poker::ComboId aces = poker::CardsToComboId(ace_spades, ace_hearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);

  poker::TrainingRange range;
  range.add(aces, 1.0f);
  range.add(kings, 2.0f);
  const poker::TrainingRangeView view(range);
  poker::TrainingRangeView scratch;

  const poker::TrainingRangeView& blocked =
      view.copy_without_mask_into(poker::CardBit(ace_spades), scratch);
  CHECK(blocked.size() == 1);
  CHECK(blocked.combo(0) == kings);

  const poker::TrainingRangeView& unblocked =
      view.copy_without_mask_into(0, scratch);
  CHECK(unblocked.size() == 2);
  CHECK(unblocked.combo(0) == aces);
  CHECK(unblocked.combo(1) == kings);
}

}  // namespace
