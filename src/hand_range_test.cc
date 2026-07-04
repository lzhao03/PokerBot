#include "src/hand_range.h"

#include <cmath>
#include <array>
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

double WeightForIndex(
    const poker::WeightedHandRange& combos, int index) {
  double weight = 0.0;
  for (size_t i = 0; i < combos.size(); ++i) {
    if (poker::HandRange::combo_to_index(combos.combos[i]) == index) {
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
    Expect(combos.combos[i] < poker::kComboCount,
           "weighted combos should be exact two-card combos");
    for (size_t j = i + 1; j < combos.size(); ++j) {
      Expect(combos.combos[i] != combos.combos[j],
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

  poker::ComboId aces_combo = poker::HandRange::index_to_combo(aces);
  poker::WeightedHandRange combos = range.get_all_weighted_combos();
  Expect(std::abs(range.get_total_weight() - 4.0) < 0.000001,
         "updating a hand should replace its old total weight");
  Expect(std::abs(WeightForIndex(combos, aces) - 3.0) < 0.000001,
         "updated class should use the replacement weight");
  Expect(std::abs(WeightForIndex(combos, kings) - 1.0) < 0.000001,
         "other class should keep its weight");
  Expect(std::abs(range.get_probability(aces_combo) - 0.125) < 0.000001,
         "class weight should be split across exact combos");
}

void CheckAddHandPreservesExactCombo() {
  poker::HandRange range;
  poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  poker::ComboId reversed =
      MakeCombo(14, poker::SuitKind::kHearts, 14, poker::SuitKind::kSpades);

  range.add_combo(aces, 1.0);

  poker::WeightedHandRange combos = range.get_all_weighted_combos();
  Expect(combos.size() == 1, "exact hand should stay one combo");
  Expect(combos.combos[0] == aces, "exact combo should be preserved");
  Expect(std::abs(combos.weights[0] - 1.0) < 0.000001,
         "exact combo should keep its weight");
  Expect(std::abs(range.get_probability(aces) - 1.0) < 0.000001,
         "exact combo should have the full range probability");

  range.add_combo(reversed, 3.0);
  combos = range.get_all_weighted_combos();
  Expect(combos.size() == 1, "same exact cards should replace existing combo");
  Expect(std::abs(range.get_total_weight() - 3.0) < 0.000001,
         "exact hand replacement should update total weight");
  Expect(std::abs(combos.weights[0] - 3.0) < 0.000001,
         "exact hand replacement should update combo weight");
}

void CheckWeightedComboInvariants() {
  poker::HandRange range;
  poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  int aces_index = poker::HandRange::string_to_index("AA");
  int ace_king_suited_index = poker::HandRange::string_to_index("AKs");

  range.add_hand_by_index(aces_index, 6.0);
  range.add_hand_by_index(ace_king_suited_index, 4.0);
  range.add_combo(aces, 2.0);

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
  poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kClubs, 13, poker::SuitKind::kDiamonds);

  combos.add(aces, 2.0);
  combos.add(kings, 3.0);

  Expect(combos.masks.size() == combos.size(),
         "weighted combo masks should match combo count");
  Expect(combos.masks[0] == poker::ComboMask(aces),
         "weighted combo should cache its hand mask");
  const poker::ComboInfo& aces_info = poker::GetComboInfo(aces);
  Expect((combos.masks[0] & poker::CardBit(aces_info.card0)) != 0,
         "weighted combo mask should include first card");
  Expect((combos.masks[0] & poker::CardBit(aces_info.card1)) != 0,
         "weighted combo mask should include second card");
  Expect((combos.masks[0] & combos.masks[1]) == 0,
         "disjoint hands should have disjoint masks");

  poker::WeightedHandRangeView view(combos);
  Expect(view.mask(1) == poker::ComboMask(kings),
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
