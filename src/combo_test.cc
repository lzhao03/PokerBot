#include "src/combo.h"

#include <array>
#include <stdexcept>

namespace {

void Expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
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

    Expect(poker::CardsToComboId(combo.card0, combo.card1) == combo_id,
           "combo id should round-trip through card ids");
    Expect(combo.mask ==
               (poker::CardBit(combo.card0) | poker::CardBit(combo.card1)),
           "combo mask should match card-id mask");
    seen[combo_id] = true;
  }

  for (bool value : seen) {
    Expect(value, "all combo ids should be visited");
  }
}

void CheckHandToComboIdIgnoresCardOrder() {
  const poker::CardId ace = poker::MakeCardId(14, poker::SuitKind::kSpades);
  const poker::CardId king = poker::MakeCardId(13, poker::SuitKind::kHearts);

  Expect(poker::CardsToComboId(ace, king) == poker::CardsToComboId(king, ace),
         "combo id should ignore hand card order");
}

void CheckInvalidHandsDoNotMapToCombos() {
  const poker::CardId ace = poker::MakeCardId(14, poker::SuitKind::kSpades);
  Expect(!poker::MaybeCardsToComboId(ace, ace).has_value(),
         "duplicate-card hand should not map to combo id");
}

}  // namespace

int main() {
  CheckComboTableHasEveryUniqueCombo();
  CheckHandToComboIdIgnoresCardOrder();
  CheckInvalidHandsDoNotMapToCombos();
  return 0;
}
