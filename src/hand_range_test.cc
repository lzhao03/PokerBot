#include "src/hand_range.h"
#include "src/card.h"

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
  poker::Card first;
  first.set_rank(first_rank);
  first.set_suit(first_suit);
  *hand.add_cards() = first;

  poker::Card second;
  second.set_rank(second_rank);
  second.set_suit(second_suit);
  *hand.add_cards() = second;
  return hand;
}

bool SameHand(const poker::Hand& left, const poker::Hand& right) {
  if (left.cards_size() != 2 || right.cards_size() != 2) {
    return false;
  }

  return (poker::SameCard(left.cards(0), right.cards(0)) &&
          poker::SameCard(left.cards(1), right.cards(1))) ||
         (poker::SameCard(left.cards(0), right.cards(1)) &&
          poker::SameCard(left.cards(1), right.cards(0)));
}

double WeightForIndex(
    const poker::WeightedHandRange& combos, int index) {
  double weight = 0.0;
  for (size_t i = 0; i < combos.size(); ++i) {
    if (poker::HandRange::hand_to_index(combos.hands[i]) == index) {
      weight += combos.weights[i];
    }
  }
  return weight;
}

double TotalComboWeight(const poker::WeightedHandRange& combos) {
  double weight = 0.0;
  for (double combo_weight : combos.weights) {
    weight += combo_weight;
  }
  return weight;
}

void CheckNoDuplicateCombos(const poker::WeightedHandRange& combos) {
  for (size_t i = 0; i < combos.size(); ++i) {
    Expect(combos.hands[i].cards_size() == 2,
           "weighted combos should be exact two-card hands");
    Expect(!poker::SameCard(combos.hands[i].cards(0),
                            combos.hands[i].cards(1)),
           "weighted combos should not duplicate a card");
    for (size_t j = i + 1; j < combos.size(); ++j) {
      Expect(!SameHand(combos.hands[i], combos.hands[j]),
             "weighted combos should not contain duplicate exact hands");
    }
  }
}

void CheckUpdatingHandWeightReplacesTotal() {
  poker::HandRange range;
  int aces = poker::HandRange::string_to_index("AA");
  int kings = poker::HandRange::string_to_index("KK");

  range.add_hand_by_index(aces, 1.0);
  range.add_hand_by_index(kings, 1.0);
  range.add_hand_by_index(aces, 3.0);

  poker::Hand aces_hand = poker::HandRange::index_to_hand(aces);
  poker::WeightedHandRange combos = range.get_all_weighted_combos();
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

  poker::WeightedHandRange combos = range.get_all_weighted_combos();
  Expect(combos.size() == 1, "exact hand should stay one combo");
  Expect(SameHand(combos.hands[0], aces), "exact combo should be preserved");
  Expect(std::abs(combos.weights[0] - 1.0) < 0.000001,
         "exact combo should keep its weight");
  Expect(std::abs(range.get_probability(aces) - 1.0) < 0.000001,
         "exact combo should have the full range probability");

  range.add_hand(reversed, 3.0);
  combos = range.get_all_weighted_combos();
  Expect(combos.size() == 1, "same exact cards should replace existing combo");
  Expect(std::abs(range.get_total_weight() - 3.0) < 0.000001,
         "exact hand replacement should update total weight");
  Expect(std::abs(combos.weights[0] - 3.0) < 0.000001,
         "exact hand replacement should update combo weight");
}

void CheckWeightedComboInvariants() {
  poker::HandRange range;
  poker::Hand aces =
      MakeHand(14, poker::Suit::SPADES, 14, poker::Suit::HEARTS);
  int aces_index = poker::HandRange::string_to_index("AA");
  int ace_king_suited_index = poker::HandRange::string_to_index("AKs");

  range.add_hand_by_index(aces_index, 6.0);
  range.add_hand_by_index(ace_king_suited_index, 4.0);
  range.add_hand(aces, 2.0);

  poker::WeightedHandRange combos = range.get_all_weighted_combos();
  CheckNoDuplicateCombos(combos);
  Expect(combos.size() == 10,
         "overlapping exact and class weights should merge exact combos");
  Expect(std::abs(TotalComboWeight(combos) - range.get_total_weight()) <
             0.000001,
         "combo weights should sum to range total weight");
  Expect(std::abs(range.get_probability(aces) - 0.25) < 0.000001,
         "exact probability should include exact plus class combo weight");

  range.normalize();
  combos = range.get_all_weighted_combos();
  CheckNoDuplicateCombos(combos);
  Expect(std::abs(range.get_total_weight() - 1.0) < 0.000001,
         "normalized range total should be one");
  Expect(std::abs(TotalComboWeight(combos) - 1.0) < 0.000001,
         "normalized combo weights should sum to one");
}

void CheckWeightedComboMasks() {
  poker::WeightedHandRange combos;
  poker::Hand aces =
      MakeHand(14, poker::Suit::SPADES, 14, poker::Suit::HEARTS);
  poker::Hand kings =
      MakeHand(13, poker::Suit::CLUBS, 13, poker::Suit::DIAMONDS);

  combos.add(aces, 2.0);
  combos.add(kings, 3.0);

  Expect(combos.masks.size() == combos.size(),
         "weighted combo masks should match combo count");
  Expect(combos.masks[0] == poker::HandMask(aces),
         "weighted combo should cache its hand mask");
  Expect((combos.masks[0] & poker::CardBit(aces.cards(0))) != 0,
         "weighted combo mask should include first card");
  Expect((combos.masks[0] & poker::CardBit(aces.cards(1))) != 0,
         "weighted combo mask should include second card");
  Expect((combos.masks[0] & combos.masks[1]) == 0,
         "disjoint hands should have disjoint masks");

  poker::WeightedHandRangeView view(combos);
  Expect(view.mask(1) == poker::HandMask(kings),
         "range view should expose source hand masks");
}

}  // namespace

int main() {
  CheckUpdatingHandWeightReplacesTotal();
  CheckAddHandPreservesExactCombo();
  CheckWeightedComboInvariants();
  CheckWeightedComboMasks();
  return 0;
}
