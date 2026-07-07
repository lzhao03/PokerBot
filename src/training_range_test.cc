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

void ExpectMatchesWeightedCombos(const poker::WeightedHandRange& combos,
                                 const poker::TrainingRange& range) {
  Expect(range.active_count == combos.size(),
         "training range should preserve active combo count");
  for (size_t i = 0; i < combos.size(); ++i) {
    const poker::ComboId combo = combos.combos[i];
    Expect(range.active[i] == combo,
           "training range should preserve combo order");
    Expect(std::abs(range.weight(combo) -
                    static_cast<float>(combos.weights[i])) < 0.000001f,
           "training range should preserve combo weights");
  }
}

void CheckTrainingRangeConversionInvariants() {
  poker::HandRange hand_range;
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AA"), 6.0);
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AKs"), 4.0);
  hand_range.add_combo(
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts),
      2.0);

  const poker::WeightedHandRange& combos = hand_range.get_all_weighted_combos();
  const poker::TrainingRange range = poker::BuildTrainingRange(hand_range);
  ExpectMatchesWeightedCombos(combos, range);
  Expect(std::abs(TotalWeight(range) -
                  static_cast<float>(hand_range.get_total_weight())) <
             0.000001f,
         "training weights should sum to source total");

  poker::HandRange empty;
  Expect(poker::BuildTrainingRange(empty).empty(),
         "empty source should build an empty training range");
}

void CheckTrainingRangeViewResetInvariants() {
  poker::WeightedHandRange combos;
  const poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);
  combos.add(aces, 1.0);
  combos.add(kings, 2.0);

  const poker::TrainingRange range = poker::BuildTrainingRange(combos);
  poker::TrainingRangeView view(range);
  Expect(view.size() == 2, "all-source view should expose source combos");

  view.reset_to_filtered();
  view.add(aces, 0.25f);
  view.add(kings, 0.75f);
  Expect(view.size() == 2, "filtered view should expose added combos");

  view.reset_to_filtered();
  Expect(view.empty(), "reset should clear active combos");
  Expect(view.weights[aces] == 0.0f && view.weights[kings] == 0.0f,
         "reset should clear touched weights");

  view.add(kings, 0.5f);
  Expect(view.size() == 1 && view.combo(0) == kings &&
             std::abs(view.weight(0) - 0.5f) < 0.000001f,
         "filtered view should be reusable after reset");
}

void CheckTrainingRangeViewWithoutMask() {
  const poker::CardId ace_spades =
      poker::MakeCardId(14, poker::SuitKind::kSpades);
  const poker::CardId ace_hearts =
      poker::MakeCardId(14, poker::SuitKind::kHearts);
  const poker::ComboId aces = poker::CardsToComboId(ace_spades, ace_hearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);

  poker::WeightedHandRange combos;
  combos.add(aces, 1.0);
  combos.add(kings, 2.0);
  const poker::TrainingRange range = poker::BuildTrainingRange(combos);
  const poker::TrainingRangeView view(range);
  poker::TrainingRangeView scratch;

  const poker::TrainingRangeView& blocked =
      view.without_mask(poker::CardBit(ace_spades), scratch);
  Expect(blocked.size() == 1 && blocked.combo(0) == kings,
         "without_mask should remove overlapping combos");

  const poker::TrainingRangeView& unblocked = view.without_mask(0, scratch);
  Expect(unblocked.size() == 2 && unblocked.combo(0) == aces &&
             unblocked.combo(1) == kings,
         "without_mask should reuse scratch for a later wider filter");
}

}  // namespace

int main() {
  CheckTrainingRangeConversionInvariants();
  CheckTrainingRangeViewResetInvariants();
  CheckTrainingRangeViewWithoutMask();
  return 0;
}
