#include "src/training_range.h"

#include "doctest/doctest.h"

namespace {

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

void CheckMatchesWeightedCombos(const poker::WeightedHandRange& combos,
                                const poker::TrainingRange& range) {
  CHECK(range.active_count == combos.size());
  for (size_t i = 0; i < combos.size(); ++i) {
    const poker::ComboId combo = combos.combos[i];
    CAPTURE(i);
    CHECK(range.active[i] == combo);
    CHECK(range.weight(combo) ==
          doctest::Approx(static_cast<float>(combos.weights[i])).epsilon(1e-6));
  }
}

TEST_CASE("training range preserves weighted combos") {
  poker::HandRange hand_range;
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AA"), 6.0);
  hand_range.add_hand_by_index(poker::HandRange::string_to_index("AKs"), 4.0);
  hand_range.add_combo(
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts),
      2.0);

  const poker::WeightedHandRange& combos = hand_range.get_all_weighted_combos();
  const poker::TrainingRange range = poker::BuildTrainingRange(hand_range);
  CheckMatchesWeightedCombos(combos, range);
  CHECK(TotalWeight(range) ==
        doctest::Approx(static_cast<float>(hand_range.get_total_weight()))
            .epsilon(1e-6));

  poker::HandRange empty;
  CHECK(poker::BuildTrainingRange(empty).empty());
}

TEST_CASE("training range view can reset and reuse filtered storage") {
  poker::WeightedHandRange combos;
  const poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);
  combos.add(aces, 1.0);
  combos.add(kings, 2.0);

  const poker::TrainingRange range = poker::BuildTrainingRange(combos);
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

  poker::WeightedHandRange combos;
  combos.add(aces, 1.0);
  combos.add(kings, 2.0);
  const poker::TrainingRange range = poker::BuildTrainingRange(combos);
  const poker::TrainingRangeView view(range);
  poker::TrainingRangeView scratch;

  const poker::TrainingRangeView& blocked =
      view.without_mask(poker::CardBit(ace_spades), scratch);
  CHECK(blocked.size() == 1);
  CHECK(blocked.combo(0) == kings);

  const poker::TrainingRangeView& unblocked = view.without_mask(0, scratch);
  CHECK(unblocked.size() == 2);
  CHECK(unblocked.combo(0) == aces);
  CHECK(unblocked.combo(1) == kings);
}

}  // namespace
