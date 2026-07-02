#include "src/hand_range.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

poker::Hand MakeHand(int first_rank, poker::Suit first_suit, int second_rank,
                     poker::Suit second_suit) {
  poker::Hand hand;
  poker::Card* first = hand.add_cards();
  first->set_rank(first_rank);
  first->set_suit(first_suit);
  poker::Card* second = hand.add_cards();
  second->set_rank(second_rank);
  second->set_suit(second_suit);
  return hand;
}

bool SameCard(const poker::Card& left, const poker::Card& right) {
  return left.rank() == right.rank() && left.suit() == right.suit();
}

bool SameHand(const poker::Hand& left, const poker::Hand& right) {
  if (left.cards_size() != 2 || right.cards_size() != 2) {
    return false;
  }

  return (SameCard(left.cards(0), right.cards(0)) &&
          SameCard(left.cards(1), right.cards(1))) ||
         (SameCard(left.cards(0), right.cards(1)) &&
          SameCard(left.cards(1), right.cards(0)));
}

double WeightForIndex(
    const std::vector<std::pair<poker::Hand, double>>& combos, int index) {
  double weight = 0.0;
  for (const auto& combo : combos) {
    if (poker::HandRange::hand_to_index(combo.first) == index) {
      weight += combo.second;
    }
  }
  return weight;
}

void CheckUpdatingHandWeightReplacesTotal() {
  poker::HandRange range;
  int aces = poker::HandRange::string_to_index("AA");
  int kings = poker::HandRange::string_to_index("KK");

  range.add_hand_by_index(aces, 1.0);
  range.add_hand_by_index(kings, 1.0);
  range.add_hand_by_index(aces, 3.0);

  poker::Hand aces_hand = poker::HandRange::index_to_hand(aces);
  std::vector<std::pair<poker::Hand, double>> combos =
      range.get_all_weighted_combos();
  Expect(std::abs(range.get_total_weight() - 4.0) < 0.000001,
         "updating a hand should replace its old total weight");
  Expect(std::abs(WeightForIndex(combos, aces) - 3.0) < 0.000001,
         "updated class should use the replacement weight");
  Expect(std::abs(WeightForIndex(combos, kings) - 1.0) < 0.000001,
         "other class should keep its weight");
  Expect(std::abs(range.get_probability(aces_hand) - 0.125) < 0.000001,
         "class weight should be split across exact combos");
}

void CheckAddHandPreservesExactCombo() {
  poker::HandRange range;
  poker::Hand aces =
      MakeHand(14, poker::Suit::SPADES, 14, poker::Suit::HEARTS);
  poker::Hand reversed =
      MakeHand(14, poker::Suit::HEARTS, 14, poker::Suit::SPADES);

  range.add_hand(aces, 1.0);

  std::vector<std::pair<poker::Hand, double>> combos =
      range.get_all_weighted_combos();
  Expect(combos.size() == 1, "exact hand should stay one combo");
  Expect(SameHand(combos[0].first, aces), "exact combo should be preserved");
  Expect(std::abs(combos[0].second - 1.0) < 0.000001,
         "exact combo should keep its weight");
  Expect(std::abs(range.get_probability(aces) - 1.0) < 0.000001,
         "exact combo should have the full range probability");

  range.add_hand(reversed, 3.0);
  combos = range.get_all_weighted_combos();
  Expect(combos.size() == 1, "same exact cards should replace existing combo");
  Expect(std::abs(range.get_total_weight() - 3.0) < 0.000001,
         "exact hand replacement should update total weight");
  Expect(std::abs(combos[0].second - 3.0) < 0.000001,
         "exact hand replacement should update combo weight");
}

}  // namespace

int main() {
  CheckUpdatingHandWeightReplacesTotal();
  CheckAddHandPreservesExactCombo();
  return 0;
}
