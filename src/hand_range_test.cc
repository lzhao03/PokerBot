#include "src/hand_range.h"

#include <cmath>
#include <stdexcept>
#include <vector>

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

double TotalComboWeight(const poker::WeightedHandRange& combos) {
  double total = 0.0;
  for (double weight : combos.weights) {
    total += weight;
  }
  return total;
}

void ExpectExpandedRangeInvariants(const poker::HandRange& range) {
  const poker::WeightedHandRange& combos = range.get_all_weighted_combos();
  Expect(std::abs(TotalComboWeight(combos) - range.get_total_weight()) <
             0.000001,
         "expanded combo weights should sum to range total");
  for (size_t i = 0; i < combos.size(); ++i) {
    Expect(combos.combos[i] < poker::kComboCount,
           "expanded combo id should be valid");
    Expect(combos.masks[i] == poker::ComboMask(combos.combos[i]),
           "expanded combo mask should match combo id");
    for (size_t j = i + 1; j < combos.size(); ++j) {
      Expect(combos.combos[i] != combos.combos[j],
             "expanded range should not duplicate exact combos");
    }
  }
}

void CheckWeightedRangeInvariants() {
  poker::HandRange range;
  const int aces = poker::HandRange::string_to_index("AA");
  const int ace_king_suited = poker::HandRange::string_to_index("AKs");
  const poker::ComboId exact_aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);

  range.add_hand_by_index(aces, 6.0);
  range.add_hand_by_index(ace_king_suited, 4.0);
  range.add_combo(exact_aces, 2.0);
  ExpectExpandedRangeInvariants(range);
  Expect(range.get_all_weighted_combos().size() == 10,
         "class and exact weights should merge by exact combo");
  Expect(std::abs(range.get_probability(exact_aces) - 0.25) < 0.000001,
         "exact probability should include class and exact weights");

  const poker::ComboId reversed_aces =
      MakeCombo(14, poker::SuitKind::kHearts, 14, poker::SuitKind::kSpades);
  range.add_combo(reversed_aces, 3.0);
  ExpectExpandedRangeInvariants(range);
  Expect(std::abs(range.get_probability(exact_aces) -
                  (4.0 / range.get_total_weight())) < 0.000001,
         "adding reversed exact combo should replace exact weight");

  range.normalize();
  ExpectExpandedRangeInvariants(range);
  Expect(std::abs(range.get_total_weight() - 1.0) < 0.000001,
         "normalize should scale total weight to one");
}

void CheckStringRangesAndViews() {
  std::vector<const char*> ranges = {"AA,KK", "AKs,AQo", "QQ+"};
  for (const char* text : ranges) {
    poker::HandRange range;
    range.set_from_string(text);
    ExpectExpandedRangeInvariants(range);
  }

  poker::WeightedHandRange combos;
  const poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kClubs, 13, poker::SuitKind::kDiamonds);
  combos.add(aces, 2.0);
  combos.add(kings, 3.0);

  poker::WeightedHandRangeView view(combos);
  Expect(view.size() == combos.size(), "all-source view should expose source");
  Expect(view.mask(1) == poker::ComboMask(kings),
         "range view should expose source hand masks");
  Expect((view.mask(0) & view.mask(1)) == 0,
         "disjoint hands should have disjoint masks");
}

void CheckExplicitHandTypeParsing() {
  poker::HandRange any_ace_king;
  any_ace_king.set_from_string("AK");
  Expect(any_ace_king.get_all_weighted_combos().size() == 16,
         "unsuffixed non-pair should include suited and offsuit combos");
  Expect(poker::HandRange::string_to_index("AK") == -1,
         "unsuffixed non-pair should not map to a single hand-type index");

  poker::HandRange suited_ace_king;
  suited_ace_king.set_from_string("AKs");
  Expect(suited_ace_king.get_all_weighted_combos().size() == 4,
         "suited non-pair should include four combos");

  poker::HandRange offsuit_ace_king;
  offsuit_ace_king.set_from_string("AKo");
  Expect(offsuit_ace_king.get_all_weighted_combos().size() == 12,
         "offsuit non-pair should include twelve combos");

  poker::HandRange aces;
  aces.set_from_string("AA");
  Expect(aces.get_all_weighted_combos().size() == 6,
         "pair should include six combos");

  poker::HandRange unsupported_plus;
  unsupported_plus.set_from_string("89s+");
  Expect(unsupported_plus.get_all_weighted_combos().empty(),
         "unsupported suited-plus notation should not add hands");
}

void CheckRepresentativeComboIndexing() {
  const int aces = poker::HandRange::string_to_index("AA");
  const std::optional<poker::ComboId> representative =
      poker::HandRange::index_to_combo(aces);
  Expect(representative.has_value(),
         "valid hand-type index should have a representative combo");
  Expect(poker::HandRange::combo_to_index(*representative) == aces,
         "representative combo should map back to the same hand type");
  Expect(!poker::HandRange::index_to_combo(-1).has_value(),
         "negative hand-type index should be invalid");
  Expect(!poker::HandRange::index_to_combo(169).has_value(),
         "out-of-range hand-type index should be invalid");
}

void CheckToStringIsDeterministic() {
  const int aces = poker::HandRange::string_to_index("AA");
  const int kings = poker::HandRange::string_to_index("KK");
  const poker::ComboId exact_ace_king =
      MakeCombo(14, poker::SuitKind::kSpades, 13, poker::SuitKind::kHearts);
  const poker::ComboId exact_queen_jack =
      MakeCombo(12, poker::SuitKind::kClubs, 11, poker::SuitKind::kDiamonds);

  poker::HandRange first;
  first.add_hand_by_index(kings, 1.0);
  first.add_combo(exact_queen_jack, 1.0);
  first.add_hand_by_index(aces, 1.0);
  first.add_combo(exact_ace_king, 1.0);

  poker::HandRange second;
  second.add_combo(exact_ace_king, 1.0);
  second.add_hand_by_index(aces, 1.0);
  second.add_combo(exact_queen_jack, 1.0);
  second.add_hand_by_index(kings, 1.0);

  Expect(first.to_string() == second.to_string(),
         "range string should not depend on insertion order");
}

}  // namespace

int main() {
  CheckWeightedRangeInvariants();
  CheckStringRangesAndViews();
  CheckExplicitHandTypeParsing();
  CheckRepresentativeComboIndexing();
  CheckToStringIsDeterministic();
  return 0;
}
