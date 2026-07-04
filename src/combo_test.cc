#include "src/combo.h"

#include <array>
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

void CheckComboTableHasEveryUniqueCombo() {
  const auto& table = poker::ComboTable();
  Expect(table.size() == poker::kComboCount,
         "combo table should contain every two-card combo");

  std::array<bool, poker::kComboCount> seen = {};
  for (poker::ComboId combo_id = 0; combo_id < poker::kComboCount;
       ++combo_id) {
    const poker::ComboInfo& combo = poker::GetComboInfo(combo_id);
    Expect(combo.card0 < combo.card1,
           "combo cards should be stored in canonical order");
    Expect(combo.card1 < poker::kDeckCardCount,
           "combo card ids should be valid");

    const poker::Hand hand = poker::ComboIdToHand(combo_id);
    Expect(poker::HandToComboId(hand) == combo_id,
           "combo id should round-trip through protobuf hand");
    Expect(combo.mask == poker::HandMask(hand),
           "combo mask should match protobuf hand mask");
    seen[combo_id] = true;
  }

  for (bool value : seen) {
    Expect(value, "all combo ids should be visited");
  }
}

void CheckHandToComboIdIgnoresCardOrder() {
  const poker::Hand ace_king =
      MakeHand(14, poker::Suit::SPADES, 13, poker::Suit::HEARTS);
  const poker::Hand king_ace =
      MakeHand(13, poker::Suit::HEARTS, 14, poker::Suit::SPADES);

  Expect(poker::HandToComboId(ace_king) == poker::HandToComboId(king_ace),
         "combo id should ignore hand card order");
}

void CheckInvalidHandsDoNotMapToCombos() {
  poker::Hand duplicate =
      MakeHand(14, poker::Suit::SPADES, 14, poker::Suit::SPADES);
  Expect(!poker::MaybeHandToComboId(duplicate).has_value(),
         "duplicate-card hand should not map to combo id");

  poker::Hand one_card;
  *one_card.add_cards() = poker::MakeCard(14, poker::Suit::SPADES);
  Expect(!poker::MaybeHandToComboId(one_card).has_value(),
         "non-exact hand should not map to combo id");
}

}  // namespace

int main() {
  CheckComboTableHasEveryUniqueCombo();
  CheckHandToComboIdIgnoresCardOrder();
  CheckInvalidHandsDoNotMapToCombos();
  return 0;
}
