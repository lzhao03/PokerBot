#include "src/training_range.h"

#include <cmath>
#include <stdexcept>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

poker::Hand MakeHand(int first_rank, poker::Suit first_suit, int second_rank,
                     poker::Suit second_suit) {
  poker::Hand hand;
  *hand.add_cards() = poker::MakeCard(first_rank, first_suit);
  *hand.add_cards() = poker::MakeCard(second_rank, second_suit);
  return hand;
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
  hand_range.add_hand(
      MakeHand(14, poker::Suit::SPADES, 14, poker::Suit::HEARTS), 2.0);

  const poker::WeightedHandRange& combos = hand_range.get_all_weighted_combos();
  const poker::TrainingRange training_range =
      poker::BuildTrainingRange(hand_range);

  Expect(training_range.active_count == combos.size(),
         "training range should preserve exact combo count");
  for (size_t i = 0; i < combos.size(); ++i) {
    const poker::ComboId combo_id = poker::HandToComboId(combos.hands[i]);
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
  poker::Hand aces =
      MakeHand(14, poker::Suit::SPADES, 14, poker::Suit::HEARTS);
  poker::Hand reversed =
      MakeHand(14, poker::Suit::HEARTS, 14, poker::Suit::SPADES);
  hand_range.add_hand(aces, 1.0);
  hand_range.add_hand(reversed, 3.0);

  const poker::ComboId combo_id = poker::HandToComboId(aces);
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

}  // namespace

int main() {
  CheckTrainingRangeMatchesWeightedCombos();
  CheckExactHandReplacementSurvivesConversion();
  CheckEmptyRangeConvertsToEmptyTrainingRange();
  return 0;
}
