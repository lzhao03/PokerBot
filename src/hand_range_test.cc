#include "src/hand_range.h"

#include "doctest/doctest.h"

#include <optional>
#include <vector>

namespace {

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

void CheckExpandedRangeInvariants(const poker::HandRange& range) {
  const poker::WeightedHandRange& combos = range.get_all_weighted_combos();
  CHECK(TotalComboWeight(combos) ==
        doctest::Approx(range.get_total_weight()).epsilon(1e-6));
  for (size_t i = 0; i < combos.size(); ++i) {
    CAPTURE(i);
    CHECK(combos.combos[i] < poker::kComboCount);
    CHECK(combos.masks[i] == poker::ComboMask(combos.combos[i]));
    for (size_t j = i + 1; j < combos.size(); ++j) {
      CAPTURE(j);
      CHECK(combos.combos[i] != combos.combos[j]);
    }
  }
}

TEST_CASE("weighted range invariants hold across class and exact combos") {
  poker::HandRange range;
  const int aces = poker::HandRange::string_to_index("AA");
  const int ace_king_suited = poker::HandRange::string_to_index("AKs");
  const poker::ComboId exact_aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);

  range.add_hand_by_index(aces, 6.0);
  range.add_hand_by_index(ace_king_suited, 4.0);
  range.add_combo(exact_aces, 2.0);
  CheckExpandedRangeInvariants(range);
  CHECK(range.get_all_weighted_combos().size() == 10);
  CHECK(range.get_probability(exact_aces) == doctest::Approx(0.25));

  const poker::ComboId reversed_aces =
      MakeCombo(14, poker::SuitKind::kHearts, 14, poker::SuitKind::kSpades);
  range.add_combo(reversed_aces, 3.0);
  CheckExpandedRangeInvariants(range);
  CHECK(range.get_probability(exact_aces) ==
        doctest::Approx(4.0 / range.get_total_weight()));

  range.normalize();
  CheckExpandedRangeInvariants(range);
  CHECK(range.get_total_weight() == doctest::Approx(1.0).epsilon(1e-6));
}

TEST_CASE("string ranges and weighted views expand consistently") {
  std::vector<const char*> ranges = {"AA,KK", "AKs,AQo", "QQ+"};
  for (const char* text : ranges) {
    CAPTURE(text);
    poker::HandRange range;
    range.set_from_string(text);
    CheckExpandedRangeInvariants(range);
  }

  poker::WeightedHandRange combos;
  const poker::ComboId aces =
      MakeCombo(14, poker::SuitKind::kSpades, 14, poker::SuitKind::kHearts);
  const poker::ComboId kings =
      MakeCombo(13, poker::SuitKind::kClubs, 13, poker::SuitKind::kDiamonds);
  combos.add(aces, 2.0);
  combos.add(kings, 3.0);

  poker::WeightedHandRangeView view(combos);
  CHECK(view.size() == combos.size());
  CHECK(view.mask(1) == poker::ComboMask(kings));
  CHECK((view.mask(0) & view.mask(1)) == 0);
}

TEST_CASE("hand type parser distinguishes suitedness shape") {
  poker::HandRange any_ace_king;
  any_ace_king.set_from_string("AK");
  CHECK(any_ace_king.get_all_weighted_combos().size() == 16);
  CHECK(poker::HandRange::string_to_index("AK") == -1);

  poker::HandRange suited_ace_king;
  suited_ace_king.set_from_string("AKs");
  CHECK(suited_ace_king.get_all_weighted_combos().size() == 4);

  poker::HandRange offsuit_ace_king;
  offsuit_ace_king.set_from_string("AKo");
  CHECK(offsuit_ace_king.get_all_weighted_combos().size() == 12);

  poker::HandRange aces;
  aces.set_from_string("AA");
  CHECK(aces.get_all_weighted_combos().size() == 6);

  poker::HandRange unsupported_plus;
  unsupported_plus.set_from_string("89s+");
  CHECK(unsupported_plus.get_all_weighted_combos().empty());
}

TEST_CASE("representative hand-type combo indexes round-trip") {
  const int aces = poker::HandRange::string_to_index("AA");
  const std::optional<poker::ComboId> representative =
      poker::HandRange::index_to_combo(aces);
  REQUIRE(representative.has_value());
  CHECK(poker::HandRange::combo_to_index(*representative) == aces);
  CHECK(!poker::HandRange::index_to_combo(-1).has_value());
  CHECK(!poker::HandRange::index_to_combo(169).has_value());
}

TEST_CASE("range string output is deterministic") {
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

  CHECK(first.to_string() == second.to_string());
}

}  // namespace
