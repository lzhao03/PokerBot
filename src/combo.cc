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
          static_cast<uint8_t>(first),
          static_cast<uint8_t>(second),
          CardBit(CardFromId(first)) | CardBit(CardFromId(second)),
      };
      ++combo_index;
    }
  }
  return combos;
}

}  // namespace

uint8_t CardToId(const Card& card) {
  const int rank_index = card.rank() - 2;
  const int suit_index = static_cast<int>(card.suit()) - 1;
  if (rank_index < 0 || rank_index >= 13 || suit_index < 0 ||
      suit_index >= 4) {
    throw std::invalid_argument("Invalid card");
  }
  return static_cast<uint8_t>(suit_index * 13 + rank_index);
}

Card CardFromId(uint8_t card_id) {
  if (card_id >= kDeckCardCount) {
    throw std::invalid_argument("Invalid card id");
  }
  const int rank = 2 + card_id % 13;
  const Suit suit = static_cast<Suit>(1 + card_id / 13);
  return MakeCard(rank, suit);
}

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

std::optional<ComboId> MaybeHandToComboId(const Hand& hand) {
  if (hand.cards_size() != 2) {
    return std::nullopt;
  }

  uint8_t first = 0;
  uint8_t second = 0;
  try {
    first = CardToId(hand.cards(0));
    second = CardToId(hand.cards(1));
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }

  if (first == second) {
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

ComboId HandToComboId(const Hand& hand) {
  std::optional<ComboId> combo_id = MaybeHandToComboId(hand);
  if (!combo_id.has_value()) {
    throw std::invalid_argument("Invalid exact two-card hand");
  }
  return *combo_id;
}

Hand ComboIdToHand(ComboId combo_id) {
  const ComboInfo& combo = GetComboInfo(combo_id);
  Hand hand;
  *hand.add_cards() = CardFromId(combo.card0);
  *hand.add_cards() = CardFromId(combo.card1);
  return hand;
}

}  // namespace poker
