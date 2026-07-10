#include "src/combo.h"

#include "doctest/doctest.h"

namespace poker {
namespace {

TEST_CASE("combo encoding is a canonical bijection") {
  ComboId expected = 0;
  for (int first = 0; first < kDeckCardCount; ++first) {
    for (int second = first + 1; second < kDeckCardCount; ++second) {
      const CardId a = static_cast<CardId>(first);
      const CardId b = static_cast<CardId>(second);
      CAPTURE(first);
      CAPTURE(second);
      CAPTURE(expected);

      CHECK(CardsToComboId(a, b) == expected);
      CHECK(CardsToComboId(b, a) == expected);

      const ComboInfo& info = GetComboInfo(expected);
      CHECK(info.card0 == a);
      CHECK(info.card1 == b);
      CHECK(info.mask == (CardBit(a) | CardBit(b)));
      ++expected;
    }
  }

  CHECK(expected == kComboCount);
  CHECK(!MaybeCardsToComboId(0, 0).has_value());
  CHECK(!MaybeCardsToComboId(static_cast<CardId>(kDeckCardCount), 0)
             .has_value());
}

}  // namespace
}  // namespace poker
