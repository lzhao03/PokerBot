#include "src/combo.h"

#include "doctest/doctest.h"

#include <array>

namespace {

TEST_CASE("combo table contains every unique canonical combo") {
  const auto& table = poker::ComboTable();
  CHECK(table.size() == poker::kComboCount);

  std::array<bool, poker::kComboCount> seen = {};
  for (poker::ComboId combo_id = 0; combo_id < poker::kComboCount;
       ++combo_id) {
    CAPTURE(combo_id);
    const poker::ComboInfo& combo = poker::GetComboInfo(combo_id);
    CHECK(combo.card0 < combo.card1);
    CHECK(combo.card1 < poker::kDeckCardCount);
    CHECK(poker::CardsToComboId(combo.card0, combo.card1) == combo_id);
    CHECK(combo.mask ==
          (poker::CardBit(combo.card0) | poker::CardBit(combo.card1)));
    seen[combo_id] = true;
  }

  for (bool value : seen) {
    CHECK(value);
  }
}

TEST_CASE("combo id ignores hand card order") {
  const poker::CardId ace = poker::MakeCardId(14, poker::SuitKind::kSpades);
  const poker::CardId king = poker::MakeCardId(13, poker::SuitKind::kHearts);

  CHECK(poker::CardsToComboId(ace, king) ==
        poker::CardsToComboId(king, ace));
}

TEST_CASE("invalid hands do not map to combo ids") {
  const poker::CardId ace = poker::MakeCardId(14, poker::SuitKind::kSpades);
  CHECK(!poker::MaybeCardsToComboId(ace, ace).has_value());
}

}  // namespace
