#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "src/poker_types.h"

namespace poker {

using ComboId = uint16_t;

constexpr int kComboCount = 1326;

struct ComboInfo {
  CardId card0 = 0;
  CardId card1 = 0;
  CardMask mask = 0;
};

const std::array<ComboInfo, kComboCount>& ComboTable();
const ComboInfo& GetComboInfo(ComboId combo_id);
CardMask ComboMask(ComboId combo_id);

std::optional<ComboId> MaybeCardsToComboId(CardId first, CardId second);
ComboId CardsToComboId(CardId first, CardId second);

}  // namespace poker
