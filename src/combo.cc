#include "src/combo.h"

#include <stdexcept>
#include <utility>

namespace poker {
namespace {

std::array<ComboInfo, kComboCount> BuildComboTable() {
  std::array<ComboInfo, kComboCount> combos;
  int combo_index = 0;
  for (int first = 0; first < kDeckCardCount; ++first) {
    for (int second = first + 1; second < kDeckCardCount; ++second) {
      combos[combo_index] = {
          static_cast<CardId>(first),
          static_cast<CardId>(second),
          CardBit(static_cast<CardId>(first)) |
              CardBit(static_cast<CardId>(second)),
      };
      ++combo_index;
    }
  }
  return combos;
}

}  // namespace

const std::array<ComboInfo, kComboCount>& ComboTable() {
  static const std::array<ComboInfo, kComboCount> table = BuildComboTable();
  return table;
}

const ComboInfo& GetComboInfo(ComboId combo_id) {
  if (combo_id >= kComboCount) {
    throw std::invalid_argument("Invalid combo id");
  }
  return ComboTable()[combo_id];
}

CardMask ComboMask(ComboId combo_id) {
  return GetComboInfo(combo_id).mask;
}

std::optional<ComboId> MaybeCardsToComboId(CardId first, CardId second) {
  if (first >= kDeckCardCount || second >= kDeckCardCount || first == second) {
    return std::nullopt;
  }
  if (second < first) {
    std::swap(first, second);
  }

  ComboId combo_id = 0;
  for (int card = 0; card < first; ++card) {
    combo_id += static_cast<ComboId>(kDeckCardCount - card - 1);
  }
  combo_id += static_cast<ComboId>(second - first - 1);
  return combo_id;
}

ComboId CardsToComboId(CardId first, CardId second) {
  std::optional<ComboId> combo_id = MaybeCardsToComboId(first, second);
  if (!combo_id.has_value()) {
    throw std::invalid_argument("Invalid exact two-card combo");
  }
  return *combo_id;
}

}  // namespace poker
