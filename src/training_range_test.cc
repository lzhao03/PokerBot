#include "src/training_range.h"

#include "doctest/doctest.h"

#include "src/hand_range.h"

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
      view.without_mask(poker::CardBit(ace_spades), scratch);
  CHECK(blocked.size() == 1);
  CHECK(blocked.combo(0) == kings);

  const poker::TrainingRangeView& unblocked = view.without_mask(0, scratch);
  CHECK(unblocked.size() == 2);
  CHECK(unblocked.combo(0) == aces);
  CHECK(unblocked.combo(1) == kings);
}

}  // namespace
