#include "src/card_utils.h"

#include <random>
#include <stdexcept>

namespace poker {

std::vector<CardId> BuildDeck() {
  std::vector<CardId> deck;
  deck.reserve(kDeckCardCount);
  for (int card_id = 0; card_id < kDeckCardCount; ++card_id) {
    deck.push_back(static_cast<CardId>(card_id));
  }
  return deck;
}

int CardsForNextStreet(StreetKind street) {
  switch (street) {
    case StreetKind::kPreflop:
      return 3;
    case StreetKind::kFlop:
    case StreetKind::kTurn:
      return 1;
    case StreetKind::kRiver:
      return 0;
  }
  return 0;
}

namespace {

int AvailableCards(CardMask known_mask) {
  int count = 0;
  for (int card_id = 0; card_id < kDeckCardCount; ++card_id) {
    if ((known_mask & CardBit(static_cast<CardId>(card_id))) == 0) {
      ++count;
    }
  }
  return count;
}

}  // namespace

absl::InlinedVector<CardId, 5> SampleStreetCards(const GameState& state,
                                                CardMask known_private_cards,
                                                std::mt19937& rng) {
  const int count = CardsForNextStreet(state.street);
  if (count <= 0) {
    return {};
  }

  CardMask blocked_mask = known_private_cards | state.board_mask;
  if (AvailableCards(blocked_mask) < count) {
    throw std::runtime_error("Not enough cards to sample next street");
  }

  std::uniform_int_distribution<int> card_distribution(0, kDeckCardCount - 1);
  absl::InlinedVector<CardId, 5> sampled;
  sampled.reserve(count);
  while (static_cast<int>(sampled.size()) < count) {
    const CardId card_id = static_cast<CardId>(card_distribution(rng));
    const CardMask card_bit = CardBit(card_id);
    if ((blocked_mask & card_bit) != 0) {
      continue;
    }
    blocked_mask |= card_bit;
    sampled.push_back(card_id);
  }
  return sampled;
}

}  // namespace poker
