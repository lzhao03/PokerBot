#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "src/card.h"
#include "src/poker.pb.h"

namespace poker {

using ComboId = uint16_t;

constexpr int kDeckCardCount = 52;
constexpr int kComboCount = 1326;

struct ComboInfo {
  uint8_t card0 = 0;
  uint8_t card1 = 0;
  CardMask mask = 0;
};

uint8_t CardToId(const Card& card);
Card CardFromId(uint8_t card_id);

const std::array<ComboInfo, kComboCount>& ComboTable();
const ComboInfo& GetComboInfo(ComboId combo_id);
CardMask ComboMask(ComboId combo_id);

std::optional<ComboId> MaybeHandToComboId(const Hand& hand);
ComboId HandToComboId(const Hand& hand);
Hand ComboIdToHand(ComboId combo_id);

}  // namespace poker
