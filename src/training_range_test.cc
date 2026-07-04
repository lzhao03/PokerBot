#include "src/training_range.h"

#include <cmath>
#include <stdexcept>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

::poker::ComboId MakeCombo(int first_rank,
                           poker::SuitKind first_suit,
                           int second_rank,
                           poker::SuitKind second_suit) {
  return poker::CardsToComboId(
      poker::MakeCardId(first_rank, first_suit),
      poker::MakeCardId(second_rank, second_suit));
}

float TotalWeight(const poker::TrainingRange& range) {
  float total = 0.0f;
  for (uint16_t i = 0; i < range.active_count; ++i) {
    total += range.weights[range.active[i]];
  }
  return total;
}

void CheckTrainingRangeMatchesWeightedCombos() {
  poker::HandRange hand_range;
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AA"), 6.0);
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AKs"), 4.0);
  hand_range.add_combo(
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts),
      2.0);

  const poker::WeightedHandRange& combos = hand_range.get_all_weighted_combos();
  const poker::TrainingRange training_range =
      poker::BuildTrainingRange(hand_range);

  Expect(training_range.active_count == combos.size(),
         "training range should preserve exact combo count");
  for (size_t i = 0; i < combos.size(); ++i) {
    const poker::ComboId combo_id = combos.combos[i];
    Expect(std::abs(training_range.weight(combo_id) -
                    static_cast<float>(combos.weights[i])) < 0.000001f,
           "training range combo weight should match expanded combo weight");
  }
  Expect(std::abs(TotalWeight(training_range) -
                  static_cast<float>(hand_range.get_total_weight())) <
             0.000001f,
         "training range weights should sum to source range total");
}

void CheckExactHandReplacementSurvivesConversion() {
  poker::HandRange hand_range;
  poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  poker::ComboId reversed =
      MakeCombo(14, poker::SuitKind::kHearts, 14, poker::SuitKind::kSpades);
  hand_range.add_combo(aces, 1.0);
  hand_range.add_combo(reversed, 3.0);

  const poker::ComboId combo_id = aces;
  const poker::TrainingRange training_range =
      poker::BuildTrainingRange(hand_range);

  Expect(training_range.active_count == 1,
         "same exact cards should convert to one active combo");
  Expect(training_range.active[0] == combo_id,
         "active combo should be the exact hand combo");
  Expect(std::abs(training_range.weight(combo_id) - 3.0f) < 0.000001f,
         "training range should use replaced exact hand weight");
}

void CheckEmptyRangeConvertsToEmptyTrainingRange() {
  poker::HandRange hand_range;
  const poker::TrainingRange training_range =
      poker::BuildTrainingRange(hand_range);

  Expect(training_range.empty(), "empty range should convert to empty range");
}

void CheckTrainingRangeViewClearsTouchedWeights() {
  poker::WeightedHandRange combos;
  poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);
  combos.add(aces, 1.0);
  combos.add(kings, 2.0);

  const poker::TrainingRange training_range =
      poker::BuildTrainingRange(combos);
  const poker::ComboId aces_combo = aces;
  const poker::ComboId kings_combo = kings;

  poker::TrainingRangeView view(training_range);
  Expect(view.size() == 2, "all-source view should expose source combos");

  view.reset_to_filtered();
  view.add(aces_combo, 0.25f);
  view.add(kings_combo, 0.75f);
  Expect(view.size() == 2, "filtered view should expose added combos");

  view.reset_to_filtered();
  Expect(view.empty(), "reset filtered view should be empty");
  Expect(view.weights[aces_combo] == 0.0f,
         "reset should clear previous aces weight");
  Expect(view.weights[kings_combo] == 0.0f,
         "reset should clear previous kings weight");

  view.add(kings_combo, 0.5f);
  Expect(view.size() == 1, "filtered view should reuse after reset");
  Expect(view.combo(0) == kings_combo, "filtered view should keep new combo");
  Expect(std::abs(view.weight(0) - 0.5f) < 0.000001f,
         "filtered view should keep new weight");
}

}  // namespace

int main() {
  CheckTrainingRangeMatchesWeightedCombos();
  CheckExactHandReplacementSurvivesConversion();
  CheckEmptyRangeConvertsToEmptyTrainingRange();
  CheckTrainingRangeViewClearsTouchedWeights();
  return 0;
}
